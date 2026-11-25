"""FastAPI Scoring, Analyst Queue, Monitoring, and Model Release Governance Service."""
from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request

from src.agent.decision_agent import FraudDecisionAgent
from src.agent.explainer import FraudExplainer
from src.config import load_config, resolve_path
from src.data.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    LabelStatus,
    PromoteModelRequest,
    RollbackModelRequest,
    ScoreResponse,
    TransactionRequest,
)
from src.data.storage import StorageManager
from src.features.online_store import (
    DuplicateTransactionConflictError,
    ENRICHED_PRE_DECISION_FEATURE_NAMES,
    FeatureStoreError,
    OnlineFeatureStore,
    SCHEMA_VERSION,
    StaleEventError,
    canonical_tx_fingerprint,
)
from src.models.registry import ModelGovernanceError, ModelRegistry
from src.monitoring.drift import (
    build_operational_alerts,
    compute_drift_report,
    compute_feedback_quality,
)
from src.rules.engine import RuleEngine
from src.security import (
    SlidingWindowRateLimiter,
    pseudonymize_transaction_dict,
    resolve_expected_api_key,
)
from src.training.pipeline import run_training_pipeline
from src.training.retrain import retrain_with_feedback

VALID_INITIAL_STATE_MODES = frozenset({"empty", "operational_db", "pre_test_warmup"})


class ServiceState:
    """Thread-safe container for versioned scoring models, online feature state, governance, and persistence."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.cfg: dict[str, Any] = load_config()
        self.storage: StorageManager = StorageManager(self.cfg["paths"]["database_url"])
        self.registry: ModelRegistry = ModelRegistry(
            registry_dir=self.cfg["paths"].get("registry_dir", "models/registry"),
            active_artifacts_dir=self.cfg["paths"]["artifacts_dir"],
        )
        self.rate_limiter = SlidingWindowRateLimiter(
            max_requests_per_minute=int(self.cfg.get("security", {}).get("rate_limit_per_minute", 3000))
        )
        self.feature_store: OnlineFeatureStore | None = None
        self.agent: FraudDecisionAgent | None = None
        self.metadata: dict[str, Any] = {}
        self.reference_df: pd.DataFrame = pd.DataFrame()
        self.error_counters: dict[str, int] = {
            "duplicate_conflicts": 0,
            "stale_or_contract_rejections": 0,
            "fallback_invocations": 0,
        }

    def _build_initial_feature_store(
        self,
        mode: str,
        as_of: datetime | None = None,
    ) -> OnlineFeatureStore:
        if mode not in VALID_INITIAL_STATE_MODES:
            raise ValueError(f"Unsupported initial_state_mode='{mode}'. Expected one of {sorted(VALID_INITIAL_STATE_MODES)}")

        scoring_cfg = self.cfg.get("scoring", {})
        mule_th = int(self.cfg["rules"]["mule_in_degree_threshold"])
        sender_ooo = scoring_cfg.get("max_sender_out_of_order_seconds", 300.0)
        global_stale = scoring_cfg.get("max_global_staleness_seconds", 86400.0)
        idem_size = int(scoring_cfg.get("max_idempotency_cache_size", 50000))
        startup_ts = as_of or datetime.now(timezone.utc)

        if mode == "empty":
            return OnlineFeatureStore(
                mule_in_degree_threshold=mule_th,
                max_sender_out_of_order_sec=sender_ooo,
                max_global_staleness_sec=global_stale,
                max_idempotency_entries=idem_size,
                state_origin="empty",
                as_of_timestamp=startup_ts.isoformat(),
            )

        if mode == "operational_db":
            op_events = self.storage.get_operational_history(as_of=startup_ts)
            return OnlineFeatureStore.from_history(
                op_events,
                as_of_timestamp=startup_ts,
                mule_in_degree_threshold=mule_th,
                max_sender_out_of_order_sec=sender_ooo,
                max_global_staleness_sec=global_stale,
                max_idempotency_entries=idem_size,
                state_origin="operational_db",
            )

        artifacts_dir = resolve_path(self.cfg["paths"]["artifacts_dir"])
        warmup_path = artifacts_dir / "initial_feature_store_pre_test.joblib"
        if not warmup_path.exists():
            raise FileNotFoundError(
                f"Pre-test warm-up store not found at {warmup_path}. Run `python -m src.training.pipeline` first."
            )
        store: OnlineFeatureStore = joblib.load(warmup_path)
        if getattr(store, "state_origin", "") != "pre_test_warmup":
            raise RuntimeError(
                f"Refusing to load feature store with untrusted state_origin='{getattr(store, 'state_origin', None)}'"
            )
        store.max_sender_out_of_order_sec = sender_ooo
        store.max_global_staleness_sec = global_stale
        store.max_idempotency_entries = idem_size
        return store

    def reload_model_artifacts(self) -> None:
        """Atomically reload active model, thresholds, metadata, and reference distribution without resetting live state."""
        with self._lock:
            self.cfg = load_config()
            artifacts_dir = resolve_path(self.cfg["paths"]["artifacts_dir"])
            gbdt_path = artifacts_dir / "gbdt_model.joblib"
            if not gbdt_path.exists():
                auto_train = bool(self.cfg.get("scoring", {}).get("auto_train_on_startup", False))
                if auto_train:
                    run_training_pipeline(cfg=self.cfg)
                else:
                    raise FileNotFoundError(
                        f"Active model artifact not found at {gbdt_path}. Run `python -m src.training.pipeline` first."
                    )

            gbdt_model = joblib.load(gbdt_path)
            anomaly_detector = joblib.load(artifacts_dir / "anomaly_detector.joblib")
            background_median = joblib.load(artifacts_dir / "background_median.joblib")

            meta_path = artifacts_dir / "metadata.json"
            metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

            ref_path = artifacts_dir / "reference_distribution.parquet"
            reference_df = pd.read_parquet(ref_path) if ref_path.exists() else pd.DataFrame()

            feature_names = metadata.get("feature_names") or getattr(gbdt_model, "feature_names_", None) or list(
                ENRICHED_PRE_DECISION_FEATURE_NAMES
            )
            model_version = str(metadata.get("model_version", "legacy-unversioned"))
            schema_version = str(metadata.get("feature_schema_version", SCHEMA_VERSION))
            thresholds = metadata.get("optimized_thresholds", self.cfg["agent"]["thresholds"])
            scoring_cfg = self.cfg.get("scoring", {})

            rule_engine = RuleEngine(rules_cfg=self.cfg["rules"])
            explainer = FraudExplainer(
                ml_model=gbdt_model,
                feature_names=feature_names,
                background_median=background_median,
                llm_cfg=self.cfg.get("llm", {}),
            )
            new_agent = FraudDecisionAgent(
                rule_engine=rule_engine,
                ml_model=gbdt_model,
                anomaly_detector=anomaly_detector,
                explainer=explainer,
                weights=self.cfg["agent"]["weights"],
                thresholds=thresholds,
                feature_names=feature_names,
                model_version=model_version,
                feature_schema_version=schema_version,
                anomaly_max_standalone_score=float(self.cfg["agent"].get("anomaly_max_standalone_score", 0.14)),
                fallback_policy=scoring_cfg.get("fallback_policy", "rules_only"),
            )

            self.agent = new_agent
            self.metadata = metadata
            self.reference_df = reference_df

    def load_artifacts(
        self,
        *,
        initial_state_mode: str | None = None,
        as_of: datetime | None = None,
    ) -> None:
        """Load active model artifacts and initialize a clean, leak-free OnlineFeatureStore."""
        with self._lock:
            self.reload_model_artifacts()
            mode = initial_state_mode or self.cfg.get("scoring", {}).get("initial_state_mode", "empty")
            self.feature_store = self._build_initial_feature_store(mode=mode, as_of=as_of)

    def score(self, tx: TransactionRequest, *, shadow_override: bool | None = None) -> ScoreResponse:
        """Thread-safe, idempotent, event-order-checked scoring with fail-safe fallback & audit persistence."""
        t0 = time.perf_counter()
        with self._lock:
            if self.agent is None or self.feature_store is None:
                try:
                    self.load_artifacts()
                except FileNotFoundError as exc:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc

            assert self.agent is not None
            assert self.feature_store is not None

            tx_dict = tx.model_dump()
            tx_dict["type"] = tx.type.value
            tx_dict["channel"] = tx.channel.value if tx.channel is not None else None

            sec_cfg = self.cfg.get("security", {})
            if bool(sec_cfg.get("pseudonymize_ids", False)):
                salt = str(sec_cfg.get("pseudonymization_salt", "pilot-default-salt"))
                tx_dict = pseudonymize_transaction_dict(tx_dict, salt=salt)

            scoring_cfg = self.cfg.get("scoring", {})
            shadow_mode = (
                bool(shadow_override)
                if shadow_override is not None
                else bool(scoring_cfg.get("shadow_mode", False))
            )
            timeout_budget_ms = scoring_cfg.get("scoring_timeout_budget_ms")

            # 1. Persistent idempotency & conflict check by tx_id
            existing_rec = self.storage.get_transaction(tx.tx_id)
            if existing_rec is not None:
                existing_fp = canonical_tx_fingerprint(existing_rec)
                incoming_fp = canonical_tx_fingerprint(tx_dict)
                if existing_fp != incoming_fp:
                    self.error_counters["duplicate_conflicts"] += 1
                    raise HTTPException(
                        status_code=409,
                        detail=f"Conflicting transaction payload for already-scored tx_id='{tx.tx_id}'",
                    )
                return ScoreResponse(
                    tx_id=existing_rec["tx_id"],
                    action=existing_rec["action"],
                    effective_action=existing_rec.get("effective_action", existing_rec["action"]),
                    decision_mode=existing_rec.get("decision_mode", "ACTIVE"),
                    technical_status=existing_rec.get("technical_status", "OK"),
                    fallback_reason=existing_rec.get("fallback_reason"),
                    model_version=existing_rec.get("model_version", "unversioned"),
                    feature_schema_version=existing_rec.get("feature_schema_version", SCHEMA_VERSION),
                    risk_score=existing_rec["risk_score"],
                    ml_score=existing_rec["ml_score"],
                    rule_score=existing_rec["rule_score"],
                    anomaly_score=existing_rec["anomaly_score"],
                    triggered_rules=existing_rec["triggered_rules"],
                    top_factors=existing_rec["top_factors"],
                    explanation_summary=existing_rec["explanation_summary"],
                    latency_ms=existing_rec["latency_ms"],
                    is_duplicate=True,
                )

            # 2. Compute point-in-time pre-decision features without mutating state yet
            features: dict[str, float] | None = None
            try:
                in_mem_cached = self.feature_store.get_cached_features(tx_dict)
                if in_mem_cached is not None:
                    decision = self.agent.evaluate_transaction(
                        tx_dict,
                        in_mem_cached,
                        explain=True,
                        shadow_mode=shadow_mode,
                        timeout_budget_ms=timeout_budget_ms,
                    )
                    return ScoreResponse(**decision, is_duplicate=True)

                features = self.feature_store.compute_features(
                    tx_dict,
                    update_state=False,
                    strict_fields=True,
                )
            except DuplicateTransactionConflictError as exc:
                self.error_counters["duplicate_conflicts"] += 1
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except (StaleEventError, FeatureStoreError) as exc:
                self.error_counters["stale_or_contract_rejections"] += 1
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                # Unexpected runtime failure in feature store -> approved fail-safe policy!
                self.error_counters["fallback_invocations"] += 1
                fallback_decision = self.agent.evaluate_fallback_only(
                    tx=tx_dict,
                    features=None,
                    reason=f"feature_store_failure:{type(exc).__name__}",
                    shadow_mode=shadow_mode,
                    t0=t0,
                )
                self.storage.save_scored_transaction(tx_dict, fallback_decision, {})
                return ScoreResponse(**fallback_decision, is_duplicate=False)

            # 3. Run multi-layer decision service (with internal fail-safe fallback guard)
            decision = self.agent.evaluate_transaction(
                tx_dict,
                features,
                explain=True,
                shadow_mode=shadow_mode,
                timeout_budget_ms=timeout_budget_ms,
            )
            if decision.get("technical_status") == "DEGRADED_FALLBACK":
                self.error_counters["fallback_invocations"] += 1

            # 4. Persist decision & input snapshot to operational DB, then commit feature state atomically
            self.storage.save_scored_transaction(tx_dict, decision, features)
            self.feature_store.commit_event(tx_dict, features)
            return ScoreResponse(**decision, is_duplicate=False)


state = ServiceState()


def enforce_gateway_security(request: Request) -> None:
    """Enforce service-to-service API key auth (when configured) and sliding-window rate limiting."""
    client_ip = request.client.host if request.client else "in_process"
    if not state.rate_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    expected_key = resolve_expected_api_key(state.cfg.get("security", {}))
    if expected_key:
        supplied = request.headers.get("X-API-Key", "").strip()
        if not supplied:
            auth_hdr = request.headers.get("Authorization", "").strip()
            if auth_hdr.lower().startswith("bearer "):
                supplied = auth_hdr[7:].strip()
        if supplied != expected_key:
            raise HTTPException(status_code=401, detail="Invalid or missing service API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state.load_artifacts()
    except FileNotFoundError:
        pass
    yield


app = FastAPI(
    title="AI Fraud Detection Service API",
    description=(
        "Сервис пре-скоринга банковских переводов (Правила + Калиброванный GBDT + "
        "Вспомогательный Isolation Forest + SHAP + Аудит меток + Реестр моделей)."
    ),
    version="1.0.0-pilot",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    with state._lock:
        fs = state.feature_store
        scoring_cfg = state.cfg.get("scoring", {})
        return {
            "status": "ok" if state.agent is not None else "model_not_loaded",
            "model_loaded": state.agent is not None,
            "model_version": state.agent.model_version if state.agent else None,
            "model_backend": state.metadata.get("model_backend", "unknown"),
            "dataset_source": state.metadata.get("dataset_source", "unknown"),
            "feature_schema_version": state.agent.feature_schema_version if state.agent else SCHEMA_VERSION,
            "decision_mode": "SHADOW" if scoring_cfg.get("shadow_mode", False) else "ACTIVE",
            "fallback_policy": scoring_cfg.get("fallback_policy", "rules_only"),
            "feature_store": (
                {
                    "initialized": True,
                    **fs.get_observability_stats(),
                }
                if fs
                else {
                    "initialized": False,
                    "state_origin": None,
                    "as_of_timestamp": None,
                    "ingested_events": 0,
                }
            ),
            "thresholds": (
                {
                    "otp": state.agent.t_otp,
                    "review": state.agent.t_review,
                    "block": state.agent.t_block,
                }
                if state.agent
                else {}
            ),
            "error_counters": dict(state.error_counters),
        }


@app.post("/score", response_model=ScoreResponse, dependencies=[Depends(enforce_gateway_security)])
def score_transaction(
    tx: TransactionRequest,
    shadow: bool | None = Query(default=None, description="Optional per-request shadow pilot override"),
) -> ScoreResponse:
    # Support direct Python calls (`score_transaction(tx)`) where FastAPI passes Query default object if omitted
    shadow_val = shadow if isinstance(shadow, bool) else None
    return state.score(tx, shadow_override=shadow_val)


@app.get("/alerts", dependencies=[Depends(enforce_gateway_security)])
def list_alerts(
    action: str | None = Query(default=None, description="Comma-separated actions, e.g. BLOCK,REVIEW,OTP"),
    only_unlabeled: bool = Query(default=False),
    label_status: str | None = Query(default=None, description="Filter by fraud / not_fraud / unknown"),
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    act_str = action if isinstance(action, str) else None
    ls_str = label_status if isinstance(label_status, str) else None
    unlabeled_bool = bool(only_unlabeled) if isinstance(only_unlabeled, bool) else False
    lim_int = int(limit) if isinstance(limit, int) else 100
    actions = [a.strip().upper() for a in act_str.split(",") if a.strip()] if act_str else None
    return state.storage.get_alerts(actions=actions, only_unlabeled=unlabeled_bool, label_status=ls_str, limit=lim_int)


@app.get("/alerts/{tx_id}", dependencies=[Depends(enforce_gateway_security)])
def get_alert_detail(tx_id: str) -> dict[str, Any]:
    rec = state.storage.get_transaction(tx_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not found")
    rec["label_audit_history"] = state.storage.get_feedback_audit_trail(tx_id=tx_id, limit=50)
    return rec


@app.get("/alerts/{tx_id}/audit", dependencies=[Depends(enforce_gateway_security)])
def get_alert_audit_trail(tx_id: str) -> list[dict[str, Any]]:
    rec = state.storage.get_transaction(tx_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not found")
    return state.storage.get_feedback_audit_trail(tx_id=tx_id, limit=100)


@app.post("/feedback", response_model=FeedbackResponse, dependencies=[Depends(enforce_gateway_security)])
def submit_feedback(fb: FeedbackRequest) -> FeedbackResponse:
    assert fb.label is not None
    res = state.storage.record_feedback(
        tx_id=fb.tx_id,
        is_fraud=fb.is_fraud,
        analyst_id=fb.analyst_id,
        notes=fb.notes,
        label_status=fb.label.value,
        label_source=fb.label_source,
        labeled_at=fb.labeled_at,
    )
    if res is None:
        raise HTTPException(status_code=404, detail=f"Transaction {fb.tx_id} not found")
    labeled_at, audit_id = res
    return FeedbackResponse(
        status="recorded",
        tx_id=fb.tx_id,
        label=fb.label,
        is_fraud=fb.is_fraud,
        label_source=fb.label_source,
        labeled_at=labeled_at,
        audit_id=audit_id,
    )


@app.get("/metrics/summary", dependencies=[Depends(enforce_gateway_security)])
def get_metrics_summary() -> dict[str, Any]:
    mon_cfg = state.cfg.get("monitoring", {})
    cap_cfg = state.cfg.get("agent", {}).get("capacity", {})
    summary = state.storage.get_summary_stats(
        maturity_hours=float(mon_cfg.get("label_maturity_hours", 72.0)),
        max_daily_reviews=int(cap_cfg.get("max_daily_reviews", 150)),
    )
    recent_df = state.storage.get_recent_features_df(limit=5000)
    feedback_q = compute_feedback_quality(recent_df, costs_cfg=state.cfg.get("agent", {}).get("costs"))
    drift_info = get_drift_report()
    op_alerts = build_operational_alerts(summary, drift_info.get("report"), mon_cfg)
    fs_obs = state.feature_store.get_observability_stats() if state.feature_store else {}
    return {
        "model_version": state.agent.model_version if state.agent else None,
        "operational": summary,
        "feature_store_observability": fs_obs,
        "feedback_quality": feedback_q,
        "operational_alerts": op_alerts,
        "thresholds": state.metadata.get("optimized_thresholds", {}),
    }


@app.get("/monitoring/drift", dependencies=[Depends(enforce_gateway_security)])
def get_drift_report() -> dict[str, Any]:
    recent_df = state.storage.get_recent_features_df(limit=5000)
    monitored_cols = state.metadata.get("monitored_columns", [])
    if recent_df.empty or len(recent_df) < 20 or state.reference_df.empty:
        return {
            "source": "train_vs_test_offline_baseline",
            "report": state.metadata.get("train_vs_test_drift", {}),
        }
    report = compute_drift_report(
        reference_df=state.reference_df,
        current_df=recent_df,
        columns=monitored_cols,
        warning_threshold=state.cfg["monitoring"]["psi_warning_threshold"],
        critical_threshold=state.cfg["monitoring"]["psi_critical_threshold"],
        bins=state.cfg["monitoring"]["psi_bins"],
    )
    return {"source": "live_stream_vs_training_reference", "report": report}


@app.get("/ablations", dependencies=[Depends(enforce_gateway_security)])
def get_ablations(
    dataset_source: str | None = Query(
        default=None,
        description="Optional benchmark source: 'paysim_csv' or 'enriched_simulation'",
    ),
) -> dict[str, Any]:
    ds = dataset_source if isinstance(dataset_source, str) else None
    if ds:
        bench_path = resolve_path(state.cfg["paths"]["reports_dir"]) / f"benchmark_{ds}.json"
        if bench_path.exists():
            b_meta = json.loads(bench_path.read_text(encoding="utf-8"))
            return {
                "model_version": b_meta.get("model_version"),
                "dataset_source": b_meta.get("dataset_source"),
                "dataset_title_ru": b_meta.get("dataset_title_ru"),
                "model_backend": b_meta.get("model_backend", "unknown"),
                "dataset_summary": b_meta.get("dataset_summary", {}),
                "optimized_thresholds": b_meta.get("optimized_thresholds", {}),
                "temporal_slice_stability": b_meta.get("temporal_slice_stability", {}),
                "segmented_test_metrics": b_meta.get("segmented_test_metrics", {}),
                "error_analysis": b_meta.get("error_analysis", {}),
                "ablations": b_meta.get("ablations", []),
            }
    return {
        "model_version": state.metadata.get("model_version"),
        "dataset_source": state.metadata.get("dataset_source", "enriched_simulation"),
        "dataset_title_ru": state.metadata.get("dataset_title_ru", ""),
        "model_backend": state.metadata.get("model_backend", "unknown"),
        "dataset_summary": state.metadata.get("dataset_summary", {}),
        "optimized_thresholds": state.metadata.get("optimized_thresholds", {}),
        "temporal_slice_stability": state.metadata.get("temporal_slice_stability", {}),
        "segmented_test_metrics": state.metadata.get("segmented_test_metrics", {}),
        "error_analysis": state.metadata.get("error_analysis", {}),
        "ablations": state.metadata.get("ablations", []),
    }


@app.get("/manifests", dependencies=[Depends(enforce_gateway_security)])
def get_data_manifests() -> dict[str, Any]:
    manifests_dir = resolve_path(state.cfg["paths"].get("manifests_dir", "models/manifests"))
    out: dict[str, Any] = {}
    for name in ("paysim_csv_manifest.json", "enriched_simulation_manifest.json"):
        p = manifests_dir / name
        if p.exists():
            out[name.replace("_manifest.json", "")] = json.loads(p.read_text(encoding="utf-8"))
    return out


@app.post("/retrain", dependencies=[Depends(enforce_gateway_security)])
def trigger_retrain() -> dict[str, Any]:
    """Train and register a candidate model WITHOUT automatically publishing/reloading it."""
    res = retrain_with_feedback(cfg=state.cfg, storage=state.storage, promote=False)
    if res.get("auto_promoted"):
        state.reload_model_artifacts()
    return res


@app.get("/models/registry", dependencies=[Depends(enforce_gateway_security)])
def list_model_registry() -> dict[str, Any]:
    return state.registry.list_registry()


@app.get("/models/{version_id}/report", dependencies=[Depends(enforce_gateway_security)])
def get_model_version_report(version_id: str) -> dict[str, Any]:
    meta = state.registry.get_version_metadata(version_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Model version '{version_id}' not found")
    return {
        "version_id": version_id,
        "comparison_vs_champion": meta.get("comparison_vs_champion", {}),
        "validation_cost_metrics": meta.get("validation_cost_metrics", {}),
        "test_metrics": meta.get("baseline_test_metrics") or meta.get("post_retrain_test_metrics"),
        "calibration_diagnostics": meta.get("calibration_diagnostics", {}),
        "optimized_thresholds": meta.get("optimized_thresholds", {}),
        "latency_benchmark_ms": meta.get("latency_benchmark_ms", {}),
    }


@app.post("/models/{version_id}/promote", dependencies=[Depends(enforce_gateway_security)])
def promote_model_version(version_id: str, req: PromoteModelRequest) -> dict[str, Any]:
    try:
        entry = state.registry.promote_version(
            version_id,
            approved_by=req.approved_by,
            reason=req.reason,
            override_guardrail=req.override_guardrail,
            sync_global_active=True,
        )
        state.reload_model_artifacts()
        return {"status": "promoted", "active_version": version_id, "entry": entry}
    except ModelGovernanceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/models/rollback", dependencies=[Depends(enforce_gateway_security)])
def rollback_active_model(req: RollbackModelRequest) -> dict[str, Any]:
    try:
        res = state.registry.rollback(
            target_version=req.target_version,
            initiated_by=req.initiated_by,
            reason=req.reason,
        )
        state.reload_model_artifacts()
        return res
    except ModelGovernanceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
