"""Streamlit Analyst & Model Governance Dashboard for the AI Fraud Detection Service.

Capabilities aligned with Technical Specification v1.0:
- Prioritized Alert Queue with tri-state labeling (`fraud` / `not_fraud` / `unknown`), `label_source`, and immutable audit log
- Operational KPIs: p50/p95/p99 latency, daily REVIEW capacity utilization, cold-start/missingness, mature-label & selection-bias metrics
- Model Registry & Release Governance: candidate training without auto-publish, comparison reports, manual approval (`Promote`) & instant `Rollback`
- Strictly separated benchmarks & Data Manifests for Original PaySim CSV vs Enriched Project Simulation
- Pre-decision Transaction Simulator (Active & Shadow Pilot modes)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.data.storage import StorageManager
from src.models.registry import ModelGovernanceError, ModelRegistry

st.set_page_config(
    page_title="AI Fraud Detection Service — Аналитик и Управление Релизами",
    page_icon="🛡️",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("FRAUD_API_KEY", "")
cfg = load_config()
storage = StorageManager(cfg["paths"]["database_url"])
registry = ModelRegistry(
    registry_dir=cfg["paths"].get("registry_dir", "models/registry"),
    active_artifacts_dir=cfg["paths"]["artifacts_dir"],
)


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def api_available() -> bool:
    try:
        r = httpx.get(f"{API_URL}/health", headers=_headers(), timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def load_metadata() -> dict[str, Any]:
    meta_path = resolve_path(cfg["paths"]["artifacts_dir"]) / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def load_benchmark(dataset_source: str) -> dict[str, Any]:
    p = resolve_path(cfg["paths"]["reports_dir"]) / f"benchmark_{dataset_source}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    meta = load_metadata()
    if meta.get("dataset_source") == dataset_source:
        return meta
    return {}


def load_manifests() -> dict[str, Any]:
    m_dir = resolve_path(cfg["paths"].get("manifests_dir", "models/manifests"))
    out: dict[str, Any] = {}
    for name in ("paysim_csv_manifest.json", "enriched_simulation_manifest.json"):
        fp = m_dir / name
        if fp.exists():
            out[name.replace("_manifest.json", "")] = json.loads(fp.read_text(encoding="utf-8"))
    return out


def _format_label_ru(status: str | None, legacy_bool: bool | None) -> str:
    if status == "fraud" or (status is None and legacy_bool is True):
        return "Мошенничество (fraud)"
    if status == "not_fraud" or (status is None and legacy_bool is False):
        return "Легитимный (not_fraud)"
    if status == "unknown":
        return "На проверке / не созрела (unknown)"
    return "Не размечен"


def main() -> None:
    st.title("🛡️ Сервис антифрод-скоринга банковских переводов")
    st.caption(
        "Пре-скоринг до проводки: Правила → Калиброванный GBDT + Вспомогательный Isolation Forest → "
        "Политика стоимости и лимита аналитиков → Аудит меток и контролируемый выпуск моделей"
    )

    is_api_up = api_available()
    meta = load_metadata()
    with st.sidebar:
        st.header("Состояние контура")
        if is_api_up:
            st.success(f"FastAPI подключён ({API_URL})")
        else:
            st.warning("FastAPI не запущен — локальный режим (БД + реестр артефактов)")

        if meta:
            st.info(
                f"**Активная версия:** `{meta.get('model_version', 'legacy')}`\n\n"
                f"**Бэкенд:** `{meta.get('model_backend', 'unknown')}`\n\n"
                f"**Источник обучения:** `{meta.get('dataset_source', 'enriched_simulation')}`\n\n"
                f"**Схема признаков:** `{meta.get('feature_schema_version', '2.0.0')}` ({len(meta.get('feature_names', []))} фич)"
            )
            th = meta.get("optimized_thresholds", {})
            if th:
                st.markdown(
                    f"**Пороги решения (предложение пилота):**\n"
                    f"- `OTP` ≥ `{th.get('otp', 0.3):.3f}`\n"
                    f"- `REVIEW` ≥ `{th.get('review', 0.55):.3f}`\n"
                    f"- `BLOCK` ≥ `{th.get('block', 0.8):.3f}`"
                )
        if st.button("🔄 Обновить экран"):
            st.rerun()

    tab_queue, tab_monitor, tab_registry, tab_bench, tab_sim = st.tabs([
        "🚨 Очередь аналитика и аудит",
        "📈 Мониторинг, SLA и зрелость меток",
        "🗂️ Реестр моделей и выпуск",
        "🧪 Бенчмарки и Data Manifests",
        "⚡ Пре-скоринг перевода (Active/Shadow)",
    ])

    # =========================================================================
    # TAB 1: ANALYST QUEUE, CASE CARD & TRI-STATE FEEDBACK AUDIT
    # =========================================================================
    with tab_queue:
        mon_cfg = cfg.get("monitoring", {})
        cap_cfg = cfg.get("agent", {}).get("capacity", {})
        stats = storage.get_summary_stats(
            maturity_hours=float(mon_cfg.get("label_maturity_hours", 72.0)),
            max_daily_reviews=int(cap_cfg.get("max_daily_reviews", 150)),
        )
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Всего проверено", f"{stats['total_scored']:,}")
        c2.metric("REVIEW за 24ч / лимит", f"{stats['reviews_last_24h']} / {stats['daily_review_capacity']}")
        c3.metric("Задержка p50 / p95", f"{stats['p50_latency_ms']:.1f} / {stats['p95_latency_ms']:.1f} мс")
        c4.metric("Зрелых меток (F/NF)", f"{stats['labeled_count']} ({stats['confirmed_fraud_count']} fraud)")
        c5.metric("Статус unknown / зреют", f"{stats['unknown_label_count']} / {stats['immature_unlabeled_count']}")
        c6.metric("Fail-safe срабатываний", f"{stats['degraded_fallback_count']}")

        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            selected_actions = st.multiselect(
                "Фильтр по рекомендуемому действию",
                options=["BLOCK", "REVIEW", "OTP", "ALLOW"],
                default=["BLOCK", "REVIEW", "OTP"],
            )
        with col_f2:
            only_unlabeled = st.checkbox("Только неразмеченные или `unknown`", value=False)

        alerts = storage.get_alerts(actions=selected_actions or None, only_unlabeled=only_unlabeled, limit=200)
        if not alerts:
            st.info(
                "Очередь пуста. Запустите поток (`python -m src.stream.replay --in-process`) "
                "или проверьте перевод во вкладке «⚡ Пре-скоринг перевода»."
            )
        else:
            df_alerts = pd.DataFrame([
                {
                    "tx_id": a["tx_id"],
                    "action": a["action"],
                    "effective_action": a.get("effective_action", a["action"]),
                    "mode": a.get("decision_mode", "ACTIVE"),
                    "status": a.get("technical_status", "OK"),
                    "risk_score": a["risk_score"],
                    "ml_score": a["ml_score"],
                    "rule_score": a["rule_score"],
                    "type": a["type"],
                    "amount": a["amount"],
                    "model_version": a.get("model_version", "unversioned"),
                    "label": _format_label_ru(a.get("label_status"), a.get("analyst_label")),
                    "label_source": a.get("label_source") or "—",
                }
                for a in alerts
            ])
            st.dataframe(df_alerts, use_container_width=True, height=260)

            st.divider()
            st.subheader("🔍 Карточка кейса и журнал изменений метки")
            chosen_id = st.selectbox("Выберите транзакцию для разбора", options=[a["tx_id"] for a in alerts])
            case = next(a for a in alerts if a["tx_id"] == chosen_id)

            left, right = st.columns([1.15, 1.0])
            with left:
                st.markdown(
                    f"### `{case['tx_id']}` — Решение: **{case['action']}** "
                    f"(исполнено: `{case.get('effective_action', case['action'])}`, режим: `{case.get('decision_mode', 'ACTIVE')}`)"
                )
                st.caption(
                    f"Версия модели: `{case.get('model_version')}` | Схема: `{case.get('feature_schema_version')}` | "
                    f"Тех. статус: `{case.get('technical_status', 'OK')}`"
                )
                st.write(case["explanation_summary"])

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Итоговый риск", f"{case['risk_score']:.3f}")
                m2.metric("ML (калибр.)", f"{case['ml_score']:.3f}")
                m3.metric("Правила", f"{case['rule_score']:.3f}")
                m4.metric("Аномалия (вспом.)", f"{case['anomaly_score']:.3f}")

                st.markdown("**Входной снимок до решения:**")
                st.json({
                    "type": case["type"],
                    "amount": case["amount"],
                    "nameOrig": case["nameOrig"],
                    "oldbalanceOrg": case.get("oldbalanceOrg"),
                    "nameDest": case["nameDest"],
                    "device_id": case.get("device_id"),
                    "channel": case.get("channel"),
                    "ip_country": case.get("ip_country"),
                    "timestamp": case.get("timestamp"),
                })

                if case["triggered_rules"]:
                    st.markdown("**Сработавшие правила:**")
                    for r in case["triggered_rules"]:
                        st.error(f"**[{r['severity']}] {r['rule_name']}**: {r['reason']}")

                audit_rows = storage.get_feedback_audit_trail(tx_id=case["tx_id"], limit=20)
                if audit_rows:
                    st.markdown("**Журнал аудита разметки (`feedback_audit_log`):**")
                    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)

            with right:
                st.markdown("**Факторы объяснения (SHAP):**")
                factors = case.get("top_factors", [])
                if factors:
                    f_df = pd.DataFrame(factors)
                    f_df["color"] = f_df["shap_value"].apply(lambda v: "Повышает риск" if v >= 0 else "Снижает риск")
                    fig = px.bar(
                        f_df.sort_values("shap_value"),
                        x="shap_value",
                        y="feature",
                        color="color",
                        orientation="h",
                        hover_data=["value", "description"],
                        color_discrete_map={"Повышает риск": "#d62728", "Снижает риск": "#2ca02c"},
                        title="Вклад pre-decision признаков (SHAP)",
                    )
                    fig.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.markdown("**Разметка аналитика (`fraud` / `not_fraud` / `unknown`):**")
                with st.form(key=f"fb_form_{case['tx_id']}"):
                    verdict_map = {
                        "Мошенничество (fraud)": "fraud",
                        "Легитимный перевод (not_fraud)": "not_fraud",
                        "Неизвестно / в расследовании (unknown)": "unknown",
                    }
                    verdict_ui = st.radio("Статус метки", options=list(verdict_map.keys()), horizontal=False)
                    label_source = st.selectbox(
                        "Источник метки",
                        options=["analyst_review", "customer_claim", "chargeback", "otp_abandoned", "rule_audit"],
                    )
                    analyst_id = st.text_input("ID аналитика", value=case.get("analyst_id") or "analyst_1")
                    notes = st.text_area("Комментарий расследования", value=case.get("analyst_notes") or "")
                    submitted = st.form_submit_button("💾 Записать метку в аудит")
                    if submitted:
                        chosen_lbl = verdict_map[verdict_ui]
                        if is_api_up:
                            httpx.post(
                                f"{API_URL}/feedback",
                                headers=_headers(),
                                json={
                                    "tx_id": case["tx_id"],
                                    "label": chosen_lbl,
                                    "label_source": label_source,
                                    "analyst_id": analyst_id,
                                    "notes": notes,
                                },
                                timeout=5.0,
                            )
                        else:
                            storage.record_feedback(
                                case["tx_id"],
                                analyst_id=analyst_id,
                                notes=notes,
                                label_status=chosen_lbl,
                                label_source=label_source,
                            )
                        st.success(f"Метка `{chosen_lbl}` зафиксирована с записью в журнал аудита.")
                        st.rerun()

    # =========================================================================
    # TAB 2: MONITORING, SLA, SELECTION BIAS & PSI DRIFT
    # =========================================================================
    with tab_monitor:
        st.subheader("📈 Эксплуатационный мониторинг, качество зрелых меток и PSI")
        st.info(
            "PSI — сигнал для расследования сдвига распределений, а **не** автоматический триггер публикации новой модели. "
            "Неразмеченные недавние операции учитываются как `immature` / `unknown`, а не как отрицательный класс."
        )
        summary = storage.get_summary_stats(
            maturity_hours=float(cfg.get("monitoring", {}).get("label_maturity_hours", 72.0)),
            max_daily_reviews=int(cfg.get("agent", {}).get("capacity", {}).get("max_daily_reviews", 150)),
        )
        m_a, m_b, m_c, m_d = st.columns(4)
        m_a.metric("p50 / p95 / p99 задержки", f"{summary['p50_latency_ms']} / {summary['p95_latency_ms']} / {summary['p99_latency_ms']} мс")
        m_b.metric("Загрузка очереди REVIEW (24ч)", f"{summary['review_capacity_utilization']:.0%}")
        m_c.metric("Охват метками ALLOW (selection bias)", f"{summary['label_coverage_by_action']['ALLOW']:.1%}")
        m_d.metric("Охват метками BLOCK/REVIEW", f"{max(summary['label_coverage_by_action']['BLOCK'], summary['label_coverage_by_action']['REVIEW']):.1%}")

        drift_data = meta.get("train_vs_test_drift", {})
        drift_source = "train_vs_test_offline_baseline"
        if is_api_up:
            try:
                r = httpx.get(f"{API_URL}/monitoring/drift", headers=_headers(), timeout=5.0)
                if r.status_code == 200:
                    payload = r.json()
                    drift_source = payload.get("source", drift_source)
                    drift_data = payload.get("report", drift_data)
            except Exception:
                pass

        if drift_data and drift_data.get("features"):
            st.caption(f"Источник PSI: `{drift_source}` | Статус: **{drift_data.get('overall_status', 'OK')}**")
            psi_df = pd.DataFrame(drift_data["features"])
            fig_psi = px.bar(
                psi_df,
                x="feature",
                y="psi",
                color="status",
                color_discrete_map={"OK": "#2ca02c", "WARNING": "#ff7f0e", "CRITICAL": "#d62728"},
                title="Population Stability Index (PSI) по pre-decision признакам и скорам",
            )
            fig_psi.add_hline(y=0.10, line_dash="dash", line_color="orange")
            fig_psi.add_hline(y=0.25, line_dash="dash", line_color="red")
            st.plotly_chart(fig_psi, use_container_width=True)

    # =========================================================================
    # TAB 3: MODEL REGISTRY, CANDIDATE TRAINING, PROMOTION & ROLLBACK
    # =========================================================================
    with tab_registry:
        st.subheader("🗂️ Реестр версий моделей и контролируемый выпуск")
        st.caption(
            "Ни одна модель не перезаписывает прошлые артефакты и не активируется из `/retrain` автоматически без проверки отчёта сравнения."
        )
        reg_state = registry.list_registry()
        r1, r2 = st.columns(2)
        r1.metric("Активная боевая версия", str(reg_state.get("active_version") or "нет"))
        r2.metric("Доступно версий для отката", str(len(reg_state.get("previous_active_versions", []))))

        versions_dict = reg_state.get("versions", {})
        if versions_dict:
            v_rows = []
            for vid, info in versions_dict.items():
                sm = info.get("summary_metrics", {})
                v_rows.append({
                    "version_id": vid,
                    "status": info.get("status"),
                    "dataset_source": info.get("dataset_source"),
                    "guardrail_passed": info.get("guardrail_passed"),
                    "pr_auc": sm.get("pr_auc"),
                    "net_savings": sm.get("net_savings"),
                    "daily_reviews": sm.get("daily_review_count"),
                    "created_at": info.get("created_at"),
                    "approved_by": info.get("approved_by") or "—",
                })
            st.dataframe(pd.DataFrame(v_rows), use_container_width=True)

        col_cand, col_prom, col_rb = st.columns(3)
        with col_cand:
            st.markdown("#### 1. Обучить кандидата")
            if st.button("🧪 Обучить кандидата (`/retrain`, без авто-выпуска)"):
                with st.spinner("Обучение кандидата и расчёт отчёта сравнения..."):
                    if is_api_up:
                        resp = httpx.post(f"{API_URL}/retrain", headers=_headers(), timeout=90.0)
                        st.json(resp.json())
                    else:
                        from src.training.retrain import retrain_with_feedback

                        res = retrain_with_feedback(cfg=cfg, storage=storage, promote=False)
                        st.json(res)
                st.success("Кандидат зарегистрирован в реестре! Активная модель не изменена.")

        with col_prom:
            st.markdown("#### 2. Одобрить выпуск (`Promote`)")
            cand_options = [v for v, d in versions_dict.items() if d.get("status") != "active"]
            chosen_ver = st.selectbox("Версия для выпуска", options=cand_options or ["—"])
            approver = st.text_input("Подписант выпуска", value="risk_lead_1")
            if st.button("✅ Утвердить и активировать", disabled=(not cand_options)):
                try:
                    if is_api_up:
                        r = httpx.post(
                            f"{API_URL}/models/{chosen_ver}/promote",
                            headers=_headers(),
                            json={"approved_by": approver, "reason": "Approved via Governance UI"},
                            timeout=15.0,
                        )
                        if r.status_code == 200:
                            st.success(f"Версия `{chosen_ver}` активирована!")
                        else:
                            st.error(r.text)
                    else:
                        registry.promote_version(chosen_ver, approved_by=approver, reason="Approved via Governance UI")
                        st.success(f"Версия `{chosen_ver}` активирована!")
                    st.rerun()
                except ModelGovernanceError as exc:
                    st.error(str(exc))

        with col_rb:
            st.markdown("#### 3. Экстренный откат (`Rollback`)")
            operator = st.text_input("Инициатор отката", value="oncall_engineer")
            if st.button("⏪ Откатить на предыдущую версию", disabled=(len(reg_state.get("previous_active_versions", [])) == 0)):
                try:
                    if is_api_up:
                        r = httpx.post(
                            f"{API_URL}/models/rollback",
                            headers=_headers(),
                            json={"initiated_by": operator, "reason": "Manual rollback from dashboard"},
                            timeout=15.0,
                        )
                        st.json(r.json())
                    else:
                        res_rb = registry.rollback(initiated_by=operator, reason="Manual rollback from dashboard")
                        st.json(res_rb)
                    st.rerun()
                except ModelGovernanceError as exc:
                    st.error(str(exc))

    # =========================================================================
    # TAB 4: SEPARATED BENCHMARKS (PAYSIM CSV vs ENRICHED SIMULATION) & MANIFESTS
    # =========================================================================
    with tab_bench:
        st.subheader("🧪 Раздельные бенчмарки источников данных и Data Manifests")
        st.warning(
            "Метрики на исходном PaySim CSV и на обогащённой симуляции проекта являются результатами прототипа/бенчмарка "
            "и НЕ доказывают качество на реальном банковском трафике до проведения теневого пилота."
        )
        ds_choice = st.radio(
            "Выберите источник данных для анализа",
            options=["enriched_simulation", "paysim_csv"],
            format_func=lambda x: "Обогащённая симуляция проекта (src/data/simulator.py)"
            if x == "enriched_simulation"
            else "Исходный PaySim CSV (data/paysim dataset.csv)",
            horizontal=True,
        )
        b_meta = load_benchmark(ds_choice)
        manifests = load_manifests()
        if ds_choice in manifests:
            with st.expander("📋 Data Manifest источника (SHA-256, схема, объём, ограничения)", expanded=False):
                st.json(manifests[ds_choice])

        if not b_meta or not b_meta.get("ablations"):
            st.info(f"Бенчмарк для `{ds_choice}` ещё не рассчитан. Запустите `python -m src.training.pipeline --dataset {ds_choice}`.")
        else:
            ab_df = pd.DataFrame(b_meta["ablations"])
            cols = [
                c
                for c in [
                    "variant",
                    "deployable_pre_decision",
                    "pr_auc",
                    "roc_auc",
                    "recall_at_1pct_fpr",
                    "precision_at_100",
                    "fraud_amount_capture_rate",
                    "fp_per_1000_tx",
                    "daily_review_count",
                    "net_savings",
                ]
                if c in ab_df.columns
            ]
            st.dataframe(ab_df[cols], use_container_width=True)

            stab = b_meta.get("temporal_slice_stability", {})
            if stab:
                ci = stab.get("pr_auc_bootstrap_ci95", {})
                st.markdown(
                    f"**Устойчивость на временных срезах Test:** PR-AUC 95% CI = `[{ci.get('low')}, {ci.get('high')}]` | "
                    f"Разброс по срезам (mean ± std): `{stab.get('slice_pr_auc_mean')} ± {stab.get('slice_pr_auc_std')}`"
                )
            if b_meta.get("error_analysis"):
                with st.expander("🔎 Анализ ошибок FP / FN и сегментные метрики"):
                    st.json({
                        "error_analysis": b_meta.get("error_analysis"),
                        "calibration_diagnostics": b_meta.get("calibration_diagnostics"),
                        "latency_benchmark_ms": b_meta.get("latency_benchmark_ms"),
                        "segmented_by_type": b_meta.get("segmented_test_metrics", {}).get("type", []),
                    })

    # =========================================================================
    # TAB 5: PRE-DECISION TRANSACTION SIMULATOR (ACTIVE / SHADOW)
    # =========================================================================
    with tab_sim:
        st.subheader("⚡ Проверка перевода (строго доступные до решения атрибуты)")
        with st.form("test_tx_form"):
            c1, c2, c3 = st.columns(3)
            tx_type = c1.selectbox("Тип операции", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
            amount = c2.number_input("Сумма перевода", min_value=1.0, value=75000.0, step=1000.0)
            channel = c3.selectbox("Канал (опционально)", ["MOBILE", "WEB", "API", "ATM", "NONE"])

            c4, c5, c6 = st.columns(3)
            name_orig = c4.text_input("Отправитель (nameOrig)", value="C100010")
            old_bal = c5.number_input("Баланс отправителя ДО (oldbalanceOrg)", min_value=0.0, value=76000.0, step=1000.0)
            name_dest = c6.text_input("Получатель (nameDest)", value="C_MULE_001")

            c7, c8, c9 = st.columns(3)
            device_id = c7.text_input("Устройство (пусто если нет источника)", value="DEV_ATO_99999")
            ip_country = c8.text_input("Страна IP (пусто если нет источника)", value="NL")
            shadow_flag = c9.checkbox("Теневой режим пилота (Shadow Mode)", value=False)

            hour = st.slider("Час суток UTC (0–23)", min_value=0, max_value=23, value=3)
            submit_tx = st.form_submit_button("🔍 Оценить риск до проводки")

        if submit_tx:
            now = datetime.now(timezone.utc).replace(hour=hour, minute=15, second=0, microsecond=0)
            tx_payload: dict[str, Any] = {
                "tx_id": f"TX_MANUAL_{int(datetime.now().timestamp())}",
                "step": 720 + hour,
                "timestamp": now.isoformat(),
                "type": tx_type,
                "amount": float(amount),
                "nameOrig": name_orig,
                "oldbalanceOrg": float(old_bal),
                "nameDest": name_dest,
            }
            if device_id.strip():
                tx_payload["device_id"] = device_id.strip()
            if channel != "NONE":
                tx_payload["channel"] = channel
            if ip_country.strip():
                tx_payload["ip_country"] = ip_country.strip()

            result: dict[str, Any] | None = None
            if is_api_up:
                resp = httpx.post(
                    f"{API_URL}/score",
                    headers=_headers(),
                    params={"shadow": shadow_flag},
                    json=tx_payload,
                    timeout=10.0,
                )
                if resp.status_code != 200:
                    st.error(f"Ошибка скоринга ({resp.status_code}): {resp.text}")
                else:
                    result = resp.json()
            else:
                from fastapi import HTTPException
                from src.api.app import score_transaction
                from src.data.schemas import TransactionRequest

                try:
                    result = score_transaction(TransactionRequest(**tx_payload), shadow=shadow_flag).model_dump()
                except HTTPException as exc:
                    st.error(f"Транзакция отклонена ({exc.status_code}): {exc.detail}")

            if result is not None:
                st.success(
                    f"Рекомендация: **{result['action']}** | Исполнено: **{result['effective_action']}** "
                    f"(`{result['decision_mode']}`) | Риск: **{result['risk_score']:.4f}** | "
                    f"Модель: `{result['model_version']}` ({result['latency_ms']} мс)"
                )
                st.write(result["explanation_summary"])
                st.json(result)


if __name__ == "__main__":
    main()
