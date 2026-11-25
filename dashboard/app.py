"""Streamlit Analyst Dashboard for the AI Fraud Detection Agent.

Features:
- Live KPI overview & action distribution
- Prioritized Alert Queue (BLOCK / REVIEW / OTP)
- Detailed Case Card with Risk Breakdown, SHAP bar chart, Rule hits, and Analyst Feedback form
- Drift Monitoring (PSI) & Feedback-Loop Retraining trigger
- Ablation Study & Business Cost Analysis
- Interactive Transaction Scoring Simulator
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
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path when launched via `streamlit run dashboard/app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.data.storage import StorageManager

st.set_page_config(
    page_title="AI Fraud Detection Agent — Дашборд аналитика",
    page_icon="🛡️",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
cfg = load_config()
storage = StorageManager(cfg["paths"]["database_url"])


def api_available() -> bool:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def load_metadata() -> dict[str, Any]:
    meta_path = resolve_path(cfg["paths"]["artifacts_dir"]) / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    st.title("🛡️ ИИ-агент обнаружения мошеннических переводов")
    st.caption("Многоуровневая система: Движок правил → ML-скоринг (GBDT) + Детектор аномалий (Isolation Forest) → Решающий агент + SHAP")

    is_api_up = api_available()
    with st.sidebar:
        st.header("Состояние системы")
        if is_api_up:
            st.success(f"FastAPI подключён ({API_URL})")
        else:
            st.warning("FastAPI не запущен — локальный режим (чтение из БД и артефактов)")

        meta = load_metadata()
        if meta:
            st.info(f"Модель: `{meta.get('model_backend', 'unknown')}`")
            th = meta.get("optimized_thresholds", {})
            if th:
                st.markdown(
                    f"**Пороги агента (по стоимости):**\n"
                    f"- `OTP` ≥ `{th.get('otp', 0.3):.3f}`\n"
                    f"- `REVIEW` ≥ `{th.get('review', 0.55):.3f}`\n"
                    f"- `BLOCK` ≥ `{th.get('block', 0.8):.3f}`"
                )

        if st.button("🔄 Обновить данные"):
            st.rerun()

    tab_queue, tab_monitor, tab_ablation, tab_sim = st.tabs([
        "🚨 Очередь алертов и кейсы",
        "📈 Мониторинг и дрейф (PSI)",
        "🧪 Абляции и бизнес-стоимость",
        "⚡ Тестовый скоринг транзакции",
    ])

    # =========================================================================
    # TAB 1: ALERT QUEUE & CASE CARD
    # =========================================================================
    with tab_queue:
        stats = storage.get_summary_stats()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Всего проверено", f"{stats['total_scored']:,}")
        c2.metric("Доля алертов", f"{stats['alert_rate']:.1%}")
        c3.metric("Заблокировано (₽/$)", f"{stats['blocked_amount']:,.0f}")
        c4.metric("Размечено аналитиками", f"{stats['labeled_count']}")
        c5.metric("Средняя задержка", f"{stats['avg_latency_ms']:.1f} мс")

        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            selected_actions = st.multiselect(
                "Фильтр по действию агента",
                options=["BLOCK", "REVIEW", "OTP", "ALLOW"],
                default=["BLOCK", "REVIEW", "OTP"],
            )
        with col_f2:
            only_unlabeled = st.checkbox("Только неразмеченные кейсы", value=False)

        alerts = storage.get_alerts(actions=selected_actions or None, only_unlabeled=only_unlabeled, limit=200)

        if not alerts:
            st.info(
                "Очередь алертов пока пуста. Запустите имитацию потока (`python -m src.stream.replay`) "
                "или отправьте тестовую транзакцию во вкладке «⚡ Тестовый скоринг транзакции»."
            )
        else:
            df_alerts = pd.DataFrame([
                {
                    "tx_id": a["tx_id"],
                    "action": a["action"],
                    "risk_score": a["risk_score"],
                    "ml_score": a["ml_score"],
                    "rule_score": a["rule_score"],
                    "anomaly_score": a["anomaly_score"],
                    "type": a["type"],
                    "amount": a["amount"],
                    "nameOrig": a["nameOrig"],
                    "nameDest": a["nameDest"],
                    "channel": a["channel"],
                    "analyst_label": "Мошенничество" if a["analyst_label"] is True else ("Легитимный" if a["analyst_label"] is False else "Не размечен"),
                }
                for a in alerts
            ])
            st.dataframe(df_alerts, use_container_width=True, height=280)

            st.divider()
            st.subheader("🔍 Карточка кейса")
            tx_ids = [a["tx_id"] for a in alerts]
            chosen_id = st.selectbox("Выберите транзакцию для разбора", options=tx_ids)
            case = next(a for a in alerts if a["tx_id"] == chosen_id)

            left, right = st.columns([1.2, 1.0])
            with left:
                st.markdown(f"### `{case['tx_id']}` — Решение: **{case['action']}**")
                st.write(case["explanation_summary"])

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Итоговый риск", f"{case['risk_score']:.3f}")
                m2.metric("ML-скор", f"{case['ml_score']:.3f}")
                m3.metric("Правила", f"{case['rule_score']:.3f}")
                m4.metric("Аномалия", f"{case['anomaly_score']:.3f}")

                st.markdown("**Детали перевода:**")
                st.json({
                    "type": case["type"],
                    "amount": case["amount"],
                    "sender": case["nameOrig"],
                    "recipient": case["nameDest"],
                    "device_id": case["device_id"],
                    "channel": case["channel"],
                    "ip_country": case["ip_country"],
                    "step_hour": case["step"],
                })

                if case["triggered_rules"]:
                    st.markdown("**Сработавшие правила (Level 1):**")
                    for r in case["triggered_rules"]:
                        st.error(f"**[{r['severity']}] {r['rule_name']}**: {r['reason']}")
                else:
                    st.success("Жёсткие правила не сработали.")

            with right:
                st.markdown("**Топ-факторы объяснения (SHAP):**")
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
                        title="Вклад признаков (SHAP)",
                    )
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    for f in factors:
                        st.markdown(f"- {f['description']} (`shap={f['shap_value']:+.4f}`)")

                st.divider()
                st.markdown("**Разметка аналитика (Цикл обратной связи):**")
                with st.form(key=f"fb_form_{case['tx_id']}"):
                    verdict = st.radio(
                        "Вердикт по кейсу",
                        options=["Мошенничество (Fraud)", "Легитимный перевод (Legit)"],
                        horizontal=True,
                    )
                    analyst_id = st.text_input("ID аналитика", value="analyst_1")
                    notes = st.text_area("Комментарий", value=case.get("analyst_notes") or "")
                    submitted = st.form_submit_button("💾 Сохранить разметку")
                    if submitted:
                        is_fraud = verdict.startswith("Мошенничество")
                        if is_api_up:
                            httpx.post(
                                f"{API_URL}/feedback",
                                json={
                                    "tx_id": case["tx_id"],
                                    "is_fraud": is_fraud,
                                    "analyst_id": analyst_id,
                                    "notes": notes,
                                },
                                timeout=5.0,
                            )
                        else:
                            storage.record_feedback(case["tx_id"], is_fraud, analyst_id, notes)
                        st.success("Разметка сохранена! Данные будут учтены при дообучении.")
                        st.rerun()

    # =========================================================================
    # TAB 2: MONITORING & DRIFT (PSI)
    # =========================================================================
    with tab_monitor:
        st.subheader("📉 Мониторинг дрейфа данных (PSI) и цикл дообучения")
        meta = load_metadata()
        drift_data = None
        drift_source = "train_vs_test_offline_baseline"

        if is_api_up:
            try:
                r = httpx.get(f"{API_URL}/monitoring/drift", timeout=5.0)
                if r.status_code == 200:
                    payload = r.json()
                    drift_source = payload.get("source", drift_source)
                    drift_data = payload.get("report", {})
            except Exception:
                pass
        if not drift_data:
            drift_data = meta.get("train_vs_test_drift", {})

        if drift_data and drift_data.get("features"):
            st.caption(f"Источник сравнения: `{drift_source}` | Reference: {drift_data.get('n_reference', 0)} | Current: {drift_data.get('n_current', 0)}")
            d1, d2, d3 = st.columns(3)
            d1.metric("Общий статус дрейфа", drift_data.get("overall_status", "OK"))
            d2.metric("Признаков с WARNING (PSI ≥ 0.10)", drift_data.get("n_warning", 0))
            d3.metric("Признаков с CRITICAL (PSI ≥ 0.25)", drift_data.get("n_critical", 0))

            psi_df = pd.DataFrame(drift_data["features"])
            fig_psi = px.bar(
                psi_df,
                x="feature",
                y="psi",
                color="status",
                color_discrete_map={"OK": "#2ca02c", "WARNING": "#ff7f0e", "CRITICAL": "#d62728"},
                title="Population Stability Index (PSI) по признакам и скорам",
            )
            fig_psi.add_hline(y=0.10, line_dash="dash", line_color="orange", annotation_text="Warning (0.10)")
            fig_psi.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Critical (0.25)")
            st.plotly_chart(fig_psi, use_container_width=True)
            st.dataframe(psi_df, use_container_width=True)
        else:
            st.info("Нет данных о дрейфе. Запустите обучение (`python -m src.training.pipeline`).")

        st.divider()
        st.subheader("🔁 Цикл обратной связи и дообучение")
        if st.button("🚀 Запустить дообучение модели с учётом разметки аналитиков"):
            with st.spinner("Выполняется дообучение..."):
                if is_api_up:
                    resp = httpx.post(f"{API_URL}/retrain", timeout=60.0)
                    st.json(resp.json())
                else:
                    from src.training.retrain import retrain_with_feedback

                    res = retrain_with_feedback(cfg=cfg, storage=storage)
                    st.json(res)
            st.success("Модель успешно переобучена и обновлена!")

    # =========================================================================
    # TAB 3: ABLATIONS & BUSINESS COST
    # =========================================================================
    with tab_ablation:
        st.subheader("🧪 Сравнение моделей (Абляции) и оценка бизнес-стоимости")
        meta = load_metadata()
        ablations = meta.get("ablations", [])
        if not ablations:
            st.info("Результаты абляций не найдены. Запустите `python -m src.training.pipeline`.")
        else:
            ab_df = pd.DataFrame(ablations)
            display_cols = [
                "variant",
                "pr_auc",
                "roc_auc",
                "recall_at_1pct_fpr",
                "precision_at_100",
                "fraud_prevented_amount",
                "false_positive_cost",
                "net_savings",
                "savings_rate",
            ]
            st.dataframe(ab_df[display_cols], use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                fig_pr = px.bar(
                    ab_df,
                    x="variant",
                    y="pr_auc",
                    color="pr_auc",
                    title="PR-AUC (главная метрика качества) по вариантам",
                )
                fig_pr.update_layout(xaxis_tickangle=-25)
                st.plotly_chart(fig_pr, use_container_width=True)
            with col_b:
                fig_sav = px.bar(
                    ab_df,
                    x="variant",
                    y="net_savings",
                    color="savings_rate",
                    title="Чистая экономия (Net Savings = Предотвращённый фрод − Затраты)",
                )
                fig_sav.update_layout(xaxis_tickangle=-25)
                st.plotly_chart(fig_sav, use_container_width=True)

    # =========================================================================
    # TAB 4: INTERACTIVE TRANSACTION SIMULATOR
    # =========================================================================
    with tab_sim:
        st.subheader("⚡ Отправить тестовый перевод в ИИ-агент")
        with st.form("test_tx_form"):
            c1, c2, c3 = st.columns(3)
            tx_type = c1.selectbox("Тип операции", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
            amount = c2.number_input("Сумма перевода", min_value=1.0, value=75000.0, step=1000.0)
            channel = c3.selectbox("Канал", ["MOBILE", "WEB", "API", "ATM"])

            c4, c5, c6 = st.columns(3)
            name_orig = c4.text_input("Отправитель (nameOrig)", value="C100010")
            old_bal = c5.number_input("Баланс отправителя ДО", min_value=0.0, value=76000.0, step=1000.0)
            new_bal = c6.number_input("Баланс отправителя ПОСЛЕ", min_value=0.0, value=1000.0, step=1000.0)

            c7, c8, c9 = st.columns(3)
            name_dest = c7.text_input("Получатель (nameDest)", value="C_MULE_001")
            device_id = c8.text_input("Устройство (device_id)", value="DEV_ATO_99999")
            ip_country = c9.text_input("Страна IP", value="NL")

            hour = st.slider("Час суток (0–23)", min_value=0, max_value=23, value=3)
            submit_tx = st.form_submit_button("🔍 Проверить транзакцию")

        if submit_tx:
            now = datetime.now(timezone.utc).replace(hour=hour, minute=15, second=0, microsecond=0)
            tx_payload = {
                "tx_id": f"TX_MANUAL_{int(datetime.now().timestamp())}",
                "step": 720 + hour,
                "timestamp": now.isoformat(),
                "type": tx_type,
                "amount": float(amount),
                "nameOrig": name_orig,
                "oldbalanceOrg": float(old_bal),
                "newbalanceOrig": float(new_bal),
                "nameDest": name_dest,
                "oldbalanceDest": 0.0,
                "newbalanceDest": float(amount),
                "device_id": device_id,
                "channel": channel,
                "ip_country": ip_country,
            }
            result: dict[str, Any] | None = None
            if is_api_up:
                resp = httpx.post(f"{API_URL}/score", json=tx_payload, timeout=10.0)
                if resp.status_code != 200:
                    st.error(f"Ошибка скоринга ({resp.status_code}): {resp.text}")
                else:
                    result = resp.json()
            else:
                # Local in-process scoring fallback via thread-safe ServiceState
                from fastapi import HTTPException
                from src.api.app import score_transaction
                from src.data.schemas import TransactionRequest

                try:
                    result = score_transaction(TransactionRequest(**tx_payload)).model_dump()
                except HTTPException as exc:
                    st.error(f"Транзакция отклонена ({exc.status_code}): {exc.detail}")

            if result is not None:
                st.success(f"Решение: **{result['action']}** | Риск: **{result['risk_score']:.4f}** ({result['latency_ms']} мс)")
                st.write(result["explanation_summary"])
                st.json(result)


if __name__ == "__main__":
    main()
