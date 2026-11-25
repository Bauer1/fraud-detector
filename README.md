# 🛡️ Сервис пре-скоринга банковских переводов (AI Fraud Detection Service)

Многоуровневый сервис оценки риска банковских переводов до окончательной проводки, приведённый в соответствие с [Техническим заданием и архитектурой пилота v1.0](docs/technical_specification.md):

**Приём, схема, аутентификация и идемпотентность → Point-in-time `OnlineFeatureStore` (строго pre-decision признаки без пост-балансов) → Правила + Калиброванный GBDT + Вспомогательный Isolation Forest → Политика порогов по стоимости и лимиту аналитиков (`ALLOW / OTP / REVIEW / BLOCK`) + Теневой режим и Fail-Safe → Аудит трёхзначных меток (`fraud / not_fraud / unknown`) → Версионируемый реестр моделей (`Candidate / Promote / Rollback`) + Мониторинг PSI и SLA**.

> **Важно:** в репозитории строго разделены два синтетических источника данных — **исходный PaySim CSV** (`data/paysim dataset.csv`) и **обогащённая симуляция проекта** (`src/data/simulator.py`). Полученные на них метрики являются smoke/benchmark-результатами прототипа и не заменяют валидацию на реальном банковском потоке.

---

## 📁 Структура проекта и документации

```text
fraud-detector/
├── config/config.yaml                    # Проектные параметры пилота, матрица стоимости, лимит очереди, безопасность
├── docs/
│   ├── technical_specification.md        # ТЗ, план и целевая архитектура антифрод-модели (v1.0)
│   ├── feature_contract.md               # Контракт доступности pre-decision признаков и порядка событий (v2.0.0)
│   ├── decision_and_cost_contract.md     # Контракт решений, fail-safe и чек-лист открытых вопросов банка
│   └── runbook_and_release.md            # Runbook эксплуатации, теневого пилота, выпуска и отката моделей
├── src/
│   ├── data/                             # Схемы Pydantic, профайлер/загрузчик PaySim CSV, симулятор, БД с аудитом меток
│   ├── features/                         # OnlineFeatureStore (pre-decision контракт, cold-start и missingness метрики)
│   ├── rules/                            # Детерминированный движок правил и списков (Level 1)
│   ├── models/                           # Baseline LogReg, калиброванный GBDT, вспомогательный IsoForest, ModelRegistry
│   ├── agent/                            # Сервис решений (auxiliary cap, shadow mode, fail-safe), пороги с учётом capacity, SHAP
│   ├── evaluation/                       # PR-AUC, bootstrap CI, временные срезы, сегменты, разбор FP/FN, p50/p95/p99, абляции
│   ├── monitoring/                       # PSI-дрейф, качество зрелых меток, смещение выборки (selection bias), алерты SLA
│   ├── training/                         # Раздельное офлайн-обучение по источникам и регистрация кандидатов без перезаписи
│   ├── api/app.py                        # FastAPI-сервис скоринга, очереди аналитика и управления релизами моделей
│   ├── security.py                       # HMAC-псевдонимизация ID, редактирование логов, API-ключ и rate limiter
│   └── stream/replay.py                  # Воспроизводитель потока pre-decision транзакций (Active / Shadow)
├── dashboard/app.py                      # Streamlit-рабочее место аналитика и управления реестром моделей
├── models/
│   ├── manifests/                        # Воспроизводимые Data Manifests (SHA-256, схема, объём, ограничения)
│   ├── registry/                         # Неизменяемые версии кандидатов, отчёты сравнения и стек отката
│   └── artifacts/                        # Синхронизированный снимок текущей утверждённой активной модели
├── reports/
│   ├── project_report.md                 # Сводный отчёт верификации по обоим источникам данных
│   ├── benchmark_paysim_csv.json         # Полный протокол метрик на исходном PaySim CSV
│   └── benchmark_enriched_simulation.json# Полный протокол метрик на обогащённой симуляции проекта
└── tests/                                # Юнит- и интеграционные тесты контракта признаков, утечек, аудита и релизов
```

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Профилирование источников и раздельное обучение моделей

Обучить и верифицировать модели отдельно на **исходном PaySim CSV** и **обогащённой симуляции проекта** с формированием `models/manifests/*` и регистрацией в `models/registry/`:

```bash
python -m src.training.pipeline --dataset both --promote
```

Либо отдельно по выбранному источнику (`--dataset paysim_csv` или `--dataset enriched_simulation`).

### 3. Запуск FastAPI-сервиса

```bash
uvicorn src.api.app:app --reload --port 8000
```

Основные эндпоинты (`http://127.0.0.1:8000/docs`):
- `POST /score` — пре-скоринг перевода (`?shadow=true` для теневого пилота; возвращает `model_version`, `technical_status`, `effective_action`, факторы SHAP и сработавшие правила)
- `GET /alerts`, `GET /alerts/{tx_id}`, `GET /alerts/{tx_id}/audit` — очередь аналитика, карточка кейса и журнал изменений метки
- `POST /feedback` — разметка `fraud / not_fraud / unknown` с указанием `label_source` и комментария
- `GET /metrics/summary` — задержка `p50/p95/p99`, загрузка очереди `REVIEW`, зрелые метки, смещение выборки и операционные алерты
- `GET /monitoring/drift` — отчёт PSI по признакам и скорам
- `GET /manifests` и `GET /ablations?dataset_source=paysim_csv|enriched_simulation` — Data Manifests и раздельные бенчмарки
- `POST /retrain` — обучение **кандидата** в реестре с отчётом сравнения (**без** автоматической публикации)
- `GET /models/registry`, `GET /models/{version_id}/report`, `POST /models/{version_id}/promote`, `POST /models/rollback` — реестр кандидатов, согласование выпуска и мгновенный откат

### 4. Поток транзакций и дашборд аналитика

```bash
python -m src.stream.replay --dataset enriched_simulation --n-events 300 --delay-ms 10
streamlit run dashboard/app.py
```

### 5. Запуск тестов

```bash
python -m unittest discover -s tests -v
```

---

## 📊 Раздельные результаты бенчмарка (Pre-Decision Holdout Test)

Посттранзакционные балансы (`newbalanceOrig`, `newbalanceDest`) и поле `isFlaggedFraud` исключены из входов всех боевых вариантов. Подробный разбор срезов, сегментов, ошибок FP/FN и ограничений приведён в [reports/project_report.md](reports/project_report.md).

### А. Исходный PaySim CSV (`data/paysim dataset.csv`, 25 pre-decision признаков, без фабрикации телеметрии)

| Вариант | PR-AUC | Recall @ 1% FPR | Доля суммы фрода | FP / 1000 | REVIEW / сутки |
|---|---:|---:|---:|---:|---:|
| PaySim `isFlaggedFraud` (исходное правило CSV) | 0.0030 | 0.0000 | 0.00% | 0.00 | 0.0 |
| Rules Only | 0.0165 | 0.0000 | 98.92% | 70.17 | 0.0 |
| Baseline LogReg (All Pre-Decision) | 0.0792 | 0.2958 | 99.65% | 254.04 | 20.0 |
| **Full Decision Service (Rules + GBDT + Aux Iso)** | **0.9984** | **1.0000** | **100.00%** | **0.08** | **0.0** |

### Б. Обогащённая симуляция проекта (`src/data/simulator.py`, 32 pre-decision признака)

| Вариант | PR-AUC | Recall @ 1% FPR | Доля суммы фрода | FP / 1000 | REVIEW / сутки | Ожид. стоимость / опер. |
|---|---:|---:|---:|---:|---:|---:|
| Rules Only | 0.5926 | 0.5177 | 95.16% | 45.33 | 0.0 | 29.59 |
| Baseline LogReg (Pre-Decision Tx Only) | 0.6607 | 0.6312 | 66.70% | 35.50 | 23.8 | 159.87 |
| Baseline LogReg (All Pre-Decision) | 0.9202 | 0.9078 | 95.04% | 22.00 | 14.2 | 25.14 |
| GBDT (Pre-Decision Tx Only) | 0.9459 | 0.9291 | 99.15% | 10.33 | 7.5 | 4.95 |
| GBDT (All Pre-Decision) | **0.9908** | **0.9858** | 99.49% | 1.83 | 2.6 | 2.53 |
| **Full Decision Service (Rules + GBDT + Aux Iso)** | **0.9882** | **0.9787** | **99.66%** | **1.00** | **2.1** | **1.75** |
