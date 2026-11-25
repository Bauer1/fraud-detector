"""Immutable Model Registry with candidate comparison reports, release approval, and instant rollback.

Guarantees (Technical Specification Sections 3.7, 4, 5, 8):
1. Offline training and `/retrain` NEVER overwrite previous model artifacts. Each candidate is saved
   into its own immutable directory `models/registry/<model_version>/`.
2. Every candidate stores a comparison report against the active champion (`comparison_report.json`).
3. `/retrain` registers a `candidate` only — no model becomes active without explicit promotion
   (except initial bootstrap when no active model exists yet).
4. Instant rollback restores any prior approved model version without retraining or losing history.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import load_config, resolve_path

SYNCED_ARTIFACT_FILES = [
    "gbdt_model.joblib",
    "anomaly_detector.joblib",
    "background_median.joblib",
    "metadata.json",
    "holdout_manifest.json",
    "initial_feature_store_pre_test.joblib",
    "reference_distribution.parquet",
]


class ModelGovernanceError(RuntimeError):
    """Raised when promotion or rollback violates release governance rules."""


class ModelRegistry:
    """File-backed versioned model registry for controlled release and rollback."""

    def __init__(
        self,
        registry_dir: str | Path | None = None,
        active_artifacts_dir: str | Path | None = None,
    ):
        cfg = load_config()
        self.registry_dir = resolve_path(registry_dir or cfg["paths"].get("registry_dir", "models/registry"))
        self.active_artifacts_dir = resolve_path(active_artifacts_dir or cfg["paths"]["artifacts_dir"])
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.active_artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.registry_dir / "registry.json"
        self._ensure_index()

    def _ensure_index(self) -> None:
        if not self.index_path.exists():
            initial = {
                "active_version": None,
                "active_by_dataset": {},
                "previous_active_versions": [],
                "versions": {},
                "audit_trail": [],
            }
            self._write_index(initial)

    def _read_index(self) -> dict[str, Any]:
        self._ensure_index()
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write_index(self, data: dict[str, Any]) -> None:
        tmp_path = self.index_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(self.index_path)

    def generate_version_id(self, dataset_source: str, prefix: str = "model") -> str:
        idx = self._read_index()
        seq = len(idx.get("versions", {})) + 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        short_ds = "paysim" if dataset_source == "paysim_csv" else "sim"
        vid = f"{prefix}-{short_ds}-v{seq:03d}-{ts}"
        while (self.registry_dir / vid).exists():
            seq += 1
            vid = f"{prefix}-{short_ds}-v{seq:03d}-{ts}"
        return vid

    @staticmethod
    def build_comparison_report(
        candidate_version: str,
        candidate_meta: dict[str, Any],
        champion_version: str | None,
        champion_meta: dict[str, Any] | None,
        min_pr_auc_ratio: float = 0.95,
    ) -> dict[str, Any]:
        """Compare a candidate model against the active champion on validation/test & capacity guardrails."""
        cand_val = candidate_meta.get("validation_cost_metrics", {})
        cand_test = candidate_meta.get("baseline_test_metrics") or candidate_meta.get("post_retrain_test_metrics") or {}
        cand_pr_auc = float(cand_test.get("pr_auc") or candidate_meta.get("validation_pr_auc") or 0.0)
        cand_cap_ok = bool(cand_val.get("capacity_within_limit", True))

        if not champion_version or not champion_meta:
            return {
                "candidate_version": candidate_version,
                "champion_version": None,
                "comparison_type": "initial_baseline_registration",
                "candidate_pr_auc": round(cand_pr_auc, 4),
                "candidate_val_net_savings": cand_val.get("net_savings"),
                "candidate_capacity_within_limit": cand_cap_ok,
                "guardrail_passed": True,
                "reasons": ["Initial baseline model for dataset source; no prior champion to compare against."],
            }

        champ_val = champion_meta.get("validation_cost_metrics", {})
        champ_test = champion_meta.get("baseline_test_metrics") or champion_meta.get("post_retrain_test_metrics") or {}
        champ_pr_auc = float(champ_test.get("pr_auc") or champion_meta.get("validation_pr_auc") or 0.0)

        pr_auc_delta = round(cand_pr_auc - champ_pr_auc, 4)
        pr_auc_ratio = (cand_pr_auc / champ_pr_auc) if champ_pr_auc > 0 else 1.0
        val_savings_delta = round(
            float(cand_val.get("net_savings", 0.0)) - float(champ_val.get("net_savings", 0.0)),
            2,
        )

        reasons: list[str] = []
        guardrail_passed = True

        # Only compare PR-AUC directly if evaluated on the same dataset source and independent holdout exists
        same_dataset = candidate_meta.get("dataset_source") == champion_meta.get("dataset_source")
        has_comparable_pr = same_dataset and cand_pr_auc > 0.0 and champ_pr_auc > 0.0
        if has_comparable_pr and pr_auc_ratio < min_pr_auc_ratio:
            guardrail_passed = False
            reasons.append(
                f"Candidate PR-AUC ({cand_pr_auc:.4f}) dropped below {min_pr_auc_ratio:.0%} of champion ({champ_pr_auc:.4f})."
            )
        if not cand_cap_ok:
            guardrail_passed = False
            reasons.append("Candidate exceeds configured analyst REVIEW queue capacity on validation.")
        if not reasons:
            reasons.append("All quality and capacity guardrails passed.")

        return {
            "candidate_version": candidate_version,
            "champion_version": champion_version,
            "same_dataset_source": same_dataset,
            "candidate_pr_auc": round(cand_pr_auc, 4),
            "champion_pr_auc": round(champ_pr_auc, 4),
            "pr_auc_delta": pr_auc_delta,
            "pr_auc_ratio": round(pr_auc_ratio, 4),
            "val_net_savings_delta": val_savings_delta,
            "candidate_capacity_within_limit": cand_cap_ok,
            "guardrail_passed": guardrail_passed,
            "reasons": reasons,
        }

    def register_candidate(
        self,
        version_id: str,
        source_files: dict[str, Path],
        metadata: dict[str, Any],
        config_snapshot: dict[str, Any],
        *,
        min_pr_auc_ratio: float = 0.95,
    ) -> dict[str, Any]:
        """Store an immutable model candidate bundle and compute comparison report without promoting."""
        v_dir = self.registry_dir / version_id
        if v_dir.exists():
            raise ModelGovernanceError(f"Refusing to overwrite existing model version '{version_id}'")
        v_dir.mkdir(parents=True, exist_ok=False)

        for target_name, src_path in source_files.items():
            if src_path.exists():
                shutil.copy2(src_path, v_dir / target_name)

        cfg_bytes = json.dumps(config_snapshot, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()[:16]
        (v_dir / "config_snapshot.json").write_bytes(
            json.dumps(config_snapshot, indent=2, ensure_ascii=False, default=str).encode("utf-8")
        )

        idx = self._read_index()
        dataset_source = str(metadata.get("dataset_source", "enriched_simulation"))
        champ_ver = idx.get("active_by_dataset", {}).get(dataset_source) or idx.get("active_version")
        champ_meta = self.get_version_metadata(champ_ver) if champ_ver else None

        metadata = dict(metadata)
        metadata["model_version"] = version_id
        metadata["config_sha256_short"] = cfg_hash

        comparison = self.build_comparison_report(
            candidate_version=version_id,
            candidate_meta=metadata,
            champion_version=champ_ver,
            champion_meta=champ_meta,
            min_pr_auc_ratio=min_pr_auc_ratio,
        )
        metadata["comparison_vs_champion"] = comparison

        (v_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        (v_dir / "comparison_report.json").write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        now = datetime.now(timezone.utc).isoformat()
        test_m = metadata.get("baseline_test_metrics") or metadata.get("post_retrain_test_metrics") or {}
        entry = {
            "version_id": version_id,
            "dataset_source": dataset_source,
            "status": "candidate",
            "created_at": now,
            "promoted_at": None,
            "approved_by": None,
            "config_sha256_short": cfg_hash,
            "feature_schema_version": metadata.get("feature_schema_version"),
            "guardrail_passed": comparison["guardrail_passed"],
            "summary_metrics": {
                "pr_auc": test_m.get("pr_auc"),
                "roc_auc": test_m.get("roc_auc"),
                "net_savings": test_m.get("net_savings"),
                "daily_review_count": test_m.get("daily_review_count"),
                "optimized_thresholds": metadata.get("optimized_thresholds"),
            },
            "comparison_vs_champion": comparison,
        }
        idx["versions"][version_id] = entry
        idx["audit_trail"].append({
            "event": "REGISTER_CANDIDATE",
            "version_id": version_id,
            "dataset_source": dataset_source,
            "guardrail_passed": comparison["guardrail_passed"],
            "timestamp_utc": now,
        })
        self._write_index(idx)
        return entry

    def _sync_to_active_artifacts_dir(self, version_id: str) -> None:
        v_dir = self.registry_dir / version_id
        if not v_dir.exists():
            raise ModelGovernanceError(f"Registry directory missing for version '{version_id}'")
        for fname in SYNCED_ARTIFACT_FILES:
            src = v_dir / fname
            if src.exists():
                shutil.copy2(src, self.active_artifacts_dir / fname)

    def promote_version(
        self,
        version_id: str,
        *,
        approved_by: str,
        reason: str = "Approved after comparison report review",
        override_guardrail: bool = False,
        sync_global_active: bool = True,
    ) -> dict[str, Any]:
        """Approve and promote a candidate model to active status while preserving previous active for rollback."""
        idx = self._read_index()
        versions = idx.get("versions", {})
        if version_id not in versions:
            raise ModelGovernanceError(f"Unknown model version '{version_id}'")

        entry = versions[version_id]
        if not entry.get("guardrail_passed", False) and not override_guardrail:
            raise ModelGovernanceError(
                f"Cannot promote '{version_id}': comparison guardrails failed "
                f"({entry.get('comparison_vs_champion', {}).get('reasons')}). Pass override_guardrail=True with sign-off if intended."
            )

        now = datetime.now(timezone.utc).isoformat()
        ds = entry["dataset_source"]

        prev_global = idx.get("active_version")
        if sync_global_active and prev_global and prev_global != version_id:
            if prev_global in versions and versions[prev_global]["status"] == "active":
                versions[prev_global]["status"] = "superseded"
            prev_stack: list[str] = idx.setdefault("previous_active_versions", [])
            if prev_global in prev_stack:
                prev_stack.remove(prev_global)
            prev_stack.append(prev_global)

        prev_ds = idx.setdefault("active_by_dataset", {}).get(ds)
        if prev_ds and prev_ds != version_id and prev_ds in versions and prev_ds != idx.get("active_version"):
            if versions[prev_ds]["status"] == "active":
                versions[prev_ds]["status"] = "superseded"

        entry["status"] = "active"
        entry["promoted_at"] = now
        entry["approved_by"] = approved_by
        entry["approval_reason"] = reason
        idx["active_by_dataset"][ds] = version_id

        if sync_global_active:
            idx["active_version"] = version_id
            self._sync_to_active_artifacts_dir(version_id)

        idx["audit_trail"].append({
            "event": "PROMOTE",
            "version_id": version_id,
            "previous_version": prev_global if sync_global_active else prev_ds,
            "dataset_source": ds,
            "approved_by": approved_by,
            "reason": reason,
            "override_guardrail": override_guardrail,
            "sync_global_active": sync_global_active,
            "timestamp_utc": now,
        })
        self._write_index(idx)
        return entry

    def rollback(
        self,
        *,
        target_version: str | None = None,
        initiated_by: str,
        reason: str = "Operational rollback",
    ) -> dict[str, Any]:
        """Atomically roll back active scoring model to a previously approved version without retraining."""
        idx = self._read_index()
        versions = idx.get("versions", {})
        current_active = idx.get("active_version")
        prev_stack: list[str] = idx.get("previous_active_versions", [])

        if target_version is None:
            if not prev_stack:
                raise ModelGovernanceError("No previous active model version available for rollback")
            target = prev_stack[-1]
        else:
            target = target_version
            if target not in versions:
                raise ModelGovernanceError(f"Rollback target version '{target}' does not exist in registry")
            if versions[target].get("approved_by") is None and versions[target].get("status") == "candidate":
                raise ModelGovernanceError(
                    f"Rollback target '{target}' was never approved; use promote_version instead"
                )

        if target == current_active:
            raise ModelGovernanceError(f"Version '{target}' is already the active model")

        now = datetime.now(timezone.utc).isoformat()
        if current_active and current_active in versions:
            versions[current_active]["status"] = "rolled_back"
        if target in prev_stack:
            prev_stack.remove(target)

        versions[target]["status"] = "active"
        versions[target]["rolled_back_to_at"] = now
        ds = versions[target]["dataset_source"]
        idx["active_version"] = target
        idx.setdefault("active_by_dataset", {})[ds] = target
        self._sync_to_active_artifacts_dir(target)

        idx["audit_trail"].append({
            "event": "ROLLBACK",
            "from_version": current_active,
            "to_version": target,
            "initiated_by": initiated_by,
            "reason": reason,
            "timestamp_utc": now,
        })
        self._write_index(idx)
        return {
            "status": "rolled_back",
            "previous_active_version": current_active,
            "active_version": target,
            "initiated_by": initiated_by,
            "reason": reason,
            "timestamp_utc": now,
        }

    def list_registry(self) -> dict[str, Any]:
        return self._read_index()

    def get_version_dir(self, version_id: str) -> Path:
        p = self.registry_dir / version_id
        if not p.exists():
            raise ModelGovernanceError(f"Model version '{version_id}' not found")
        return p

    def get_version_metadata(self, version_id: str | None) -> dict[str, Any] | None:
        if not version_id:
            return None
        meta_path = self.registry_dir / version_id / "metadata.json"
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def get_comparison_report(self, version_id: str) -> dict[str, Any]:
        rep_path = self.get_version_dir(version_id) / "comparison_report.json"
        if not rep_path.exists():
            raise ModelGovernanceError(f"Comparison report not found for '{version_id}'")
        return json.loads(rep_path.read_text(encoding="utf-8"))
