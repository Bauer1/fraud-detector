# 🛡️ AI Fraud Detection Agent — ИИ-агент для обнаружения мошеннических переводов

Многоуровневая сквозная система обнаружения мошеннических переводов в реальном времени, реализованная по архитектуре: **Движок правил → ML-скоринг (GBDT) + Детектор аномалий (Isolation Forest) → Решающий агент (пороги по бизнес-стоимости) + SHAP-объяснения → FastAPI + Streamlit + Мониторинг дрейфа (PSI) + Цикл обратной связи**.

---

## 🏗️ Архитектура

```
Поток транзакций (src/stream/replay.py)
        │
        ▼
 Приём и валидация (FastAPI / Pydantic) ──► Хранилище (Parquet + SQLite/PostgreSQL)
        │
        ▼
 Конструирование признаков без утечек (OnlineFeatureStore)
  ├─ транзакционные (сумма, тип, час, канал, опустошение баланса)
  ├─ поведенческие (скользящие окна 1ч, 24ч, 7д строго по прошлым данным)
  ├─ отклонения от нормы (z-score суммы, отношение к среднему, новый получатель/устройство)
  └─ графовые (входящая степень получателя, общие устройства, дроппер-паттерны)
        │
        ▼
 Сервис скоринга (src/api/app.py)
  ├─ Level 1: Движок правил (чёрные списки, жёсткие лимиты, velocity, drain, mule)
  ├─ Level 2a: Модель с учителем (LightGBM / HistGradientBoostingClassifier)
  └─ Level 2b: Детектор аномалий (Isolation Forest на легитимных операциях)
        │
        ▼
 Level 3: Решающий агент (src/agent/decision_agent.py)
  ├─ risk = f(rules, ml_score, anomaly_score)
  ├─ действие: ALLOW / OTP / REVIEW / BLOCK (пороги оптимизированы по стоимости)
  └─ объяснение: топ-5 факторов SHAP (+ опциональное LLM-резюме)
        │
        ├──► Дашборд аналитика (Streamlit: очередь алертов, карточка кейса, разметка)
        ▼
 Цикл обратной связи (POST /feedback → POST /retrain) + Мониторинг PSI (GET /monitoring/drift)
```

---

## 📁 Структура проекта

```text
fraud-detector/
├── config/config.yaml                # Параметры симуляции, правил, моделей, стоимости и PSI
├── src/
│   ├── data/                         # Pydantic-схемы, PaySim-симулятор, SQLAlchemy-хранилище
│   ├── features/                     # OnlineFeatureStore (leak-free) + batch pipeline
│   ├── rules/                        # Движок правил (Level 1)
│   ├── models/                       # Baseline LogReg, GBDT (LightGBM/HistGBDT), Isolation Forest
│   ├── agent/                        # Решающий агент, оптимизация порогов, SHAP + LLM explainer
│   ├── evaluation/                   # PR-AUC, ROC-AUC, Recall@FPR, P@K, Cost, абляции
│   ├── monitoring/                   # PSI-дрейф признаков/скоров и качество по разметке
│   ├── training/                     # Пайплайн обучения и дообучения с обратной связью
│   ├── api/                          # FastAPI-сервис скоринга
│   └── stream/                       # Хронологический воспроизводитель потока транзакций
├── dashboard/app.py                  # Дашборд аналитика на Streamlit
├── notebooks/                        # 01_eda_and_data_simulation.ipynb, 02_modeling_and_ablations.ipynb
├── reports/project_report.md         # Полный отчёт с метриками, абляциями и рисками
├── tests/test_pipeline.py            # Юнит- и интеграционные тесты (включая проверку утечек)
├── Dockerfile & docker-compose.yml   # Контейнеризация API + Dashboard
└── requirements.txt / pyproject.toml
```

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

> Примечание: если `lightgbm` или `shap` не установлены в среде, система автоматически использует встроенные эквиваленты (`HistGradientBoostingClassifier` из scikit-learn и быструю локальную Shapley-атрибуцию), сохраняя полную работоспособность.

### 2. Обучение моделей, подбор порогов и абляции

```bash
python -m src.training.pipeline
```

Команда сгенерирует 30 000 транзакций (`data/raw/transactions.parquet`), построит 35 leak-free признаков (`data/processed/features.parquet`), обучит модели на строгом разбиении по времени (60/20/20), подберёт пороги по бизнес-стоимости, выполнит абляции и сохранит артефакты в `models/artifacts/`.

### 3. Запуск FastAPI-сервиса скоринга

```bash
uvicorn src.api.app:app --reload --port 8000
```

Документация Swagger UI доступна по адресу: `http://127.0.0.1:8000/docs`.

Основные эндпоинты:
- `POST /score` — скоринг транзакции в реальном времени с SHAP-объяснением
- `GET /alerts` — очередь алертов (фильтрация по `action` и статусу разметки)
- `GET /alerts/{tx_id}` — подробная карточка кейса
- `POST /feedback` — отправка разметки аналитика (`is_fraud`, комментарии)
- `GET /metrics/summary` — операционные метрики и качество по разметке
- `GET /monitoring/drift` — отчёт по дрейфу данных (PSI)
- `GET /ablations` — результаты абляционного исследования
- `POST /retrain` — запуск дообучения модели с учётом разметки аналитиков

### 4. Имитация потока транзакций в реальном времени

В отдельном терминале (при запущенном FastAPI):

```bash
python -m src.stream.replay --n-events 300 --delay-ms 10
```

### 5. Запуск дашборда аналитика (Streamlit)

```bash
streamlit run dashboard/app.py
```

Откройте `http://localhost:8501` для работы с очередью алертов, карточками кейсов, графиками SHAP, мониторингом PSI, абляциями и формой ручного тестирования транзакций.

### 6. Запуск через Docker Compose

```bash
docker compose up --build
```

- FastAPI: `http://localhost:8000/docs`
- Streamlit Dashboard: `http://localhost:8501`

### 7. Запуск тестов

```bash
python -m unittest discover -s tests -v
```

---

## 📊 Краткие результаты (Test Set)

Подробный анализ, матрица стоимости и разбор рисков приведены в [reports/project_report.md](reports/project_report.md).

| Вариант | PR-AUC | Recall @ 1% FPR | Precision@100 | Чистая экономия |
|---|---:|---:|---:|---:|
| Rules Only | 0.5622 | 0.5177 | 0.7400 | 2 629 769 |
| Baseline LogReg (Tx Only) | 0.8135 | 0.7660 | 0.9000 | 1 865 639 |
| GBDT (Tx Only) | 0.9660 | 0.9504 | 1.0000 | 2 811 169 |
| GBDT (Full Features) | **0.9876** | **0.9929** | **1.0000** | 2 803 598 |
| **Full AI Agent (Rules + GBDT + IsoForest)** | **0.9745** | **0.9716** | **0.9900** | **2 804 696** |
