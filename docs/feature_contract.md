# Контракт доступности признаков и порядка событий (`v2.0.0-pre-decision`)

Документ фиксирует единые правила доступности полей к моменту принятия решения, расчёта point-in-time признаков, обработки отсутствующих источников и защиты от утечек для офлайн-обучения (`src/features/engineering.py`) и онлайн-скоринга (`src/api/app.py`, `src/features/online_store.py`).

## 1. Разделение источников данных и запрет смешения схем

В проекте строго разделены два источника:
1. **Исходный PaySim CSV (`data/paysim dataset.csv`, `dataset_source="paysim_csv"`)** — содержит только поля `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`. В CSV **нет** полей `tx_id`, `timestamp`, `device_id`, `channel`, `ip_country`. Для него формируются только **25 pre-decision признаков** (`PAYSIM_PRE_DECISION_FEATURE_NAMES`) без фабрикации телеметрии устройств/каналов/IP.
2. **Обогащённая симуляция проекта (`src/data/simulator.py`, `dataset_source="enriched_simulation"`)** — синтетический поток с явными `tx_id`, UTC `timestamp`, `device_id`, `channel`, `ip_country`. Для него формируются **32 pre-decision признака** (`ENRICHED_PRE_DECISION_FEATURE_NAMES`).

## 2. Матрица доступности полей и защита от утечек (Pre-Decision Contract)

| Поле / группа признаков | Источник | Статус доступности до решения | Правило использования в модели |
|---|---|---|---|
| `tx_id` | Канал перевода / `PAYSIM_{row_idx:08d}` | Доступен до решения | Ключ идемпотентности и детерминированной сортировки; **не входит** в признаки ML |
| `timestamp` (или `step`) | Канал перевода / PaySim CSV | Доступен до решения | Приводится к `UTC`; формирует `hour_of_day`, `is_night` и границы окон `[t - W, t]` |
| `type`, `amount` | Канал перевода / PaySim CSV | Подтверждены до решения | Формируют `amount`, `log_amount`, `type_*` (`amount > 0`) |
| `nameOrig`, `nameDest` | Псевдонимы счетов | Доступны до решения | Ключи для point-in-time агрегатов и P2P-графа; **запрещены** как сырые ID в признаках ML |
| `oldbalanceOrg` | АБС / карточный процессинг | **Условно доступен до решения** (требует подтверждения владельца данных банка) | Формирует `balance_drain_ratio`; управляется флагом `feature_contract.allow_sender_pre_balance` |
| `newbalanceOrig`, `newbalanceDest` | Посттранзакционная проводка | **Исключены (риск утечки)** | Признаки `orig_balance_zero_after`, `orig_balance_error`, `dest_balance_error` **исключены** из обучения и боевого скоринга (`allow_post_tx_balances: false`) |
| `oldbalanceDest` | Счёт получателя (часто внешний банк) | **Исключён по умолчанию** | Не используется без подтверждения синхронной доступности до списания (`allow_dest_balances: false`) |
| `device_id`, `channel`, `ip_country` | Антифрод-телеметрия ДБО (только симуляция / реальный ДБО) | Доступны только при наличии телеметрии канала | Включаются в `ENRICHED_PRE_DECISION_FEATURE_NAMES`; при `None` не фабрикуются и не связывают клиентов через фиктивный `DEV_UNKNOWN` |
| `isFlaggedFraud`, `isFraud`, `chargeback`, `analyst_label` | Поздняя разметка / внешнее правило | **Строго запрещены** | Передача в онлайн-скоринг вызывает `ForbiddenLeakageFieldError` (`HTTP 422`) |

## 3. Временная зона и представление времени

- **Каноническая временная зона:** `UTC` (`datetime.timezone.utc`).
- **Наивные `datetime` (без `tzinfo`):** трактуются как `UTC`.
- **`datetime` со смещением (`+03:00`, `+05:00`, `Z`):** приводятся к `UTC` до расчёта epoch-секунд, `hour_of_day` и `is_night` (`00:00–05:59 UTC`).
- **Отсутствие `timestamp` (`None`, например в исходном PaySim CSV):** детерминированный якорь `2026-01-01T00:00:00Z + timedelta(hours=step)` со счётчиком наблюдаемости `missing_optional_counts["timestamp"]`.

## 4. Cold Start и наблюдаемость пропусков

- Для нового отправителя (`orig_past_tx_count == 0`): все скользящие окна `1h/24h/7d = 0.0`, `amount_zscore = 0.0`, `amount_to_mean_ratio = 1.0`, `is_new_recipient = 0.0`, `is_new_device = 0.0`, `time_since_last_tx_hours = 168.0`.
- `OnlineFeatureStore.get_observability_stats()` публикует в `/health` и `/metrics/summary` долю `cold_start_rate` и счётчики отсутствующих опциональных атрибутов (`oldbalanceOrg`, `device_id`, `channel`, `ip_country`, `timestamp`). В исходном PaySim CSV доля `cold_start` по `nameOrig` превышает 99.8%, поэтому основную предсказательную силу несут транзакционные признаки и агрегаты получателя (`nameDest`).

## 5. Правила порядка событий, повторов и одинакового шага (`step`)

1. **Границы скользящих окон:** замкнутый интервал `[t - W, t]` строго по ранее зафиксированным событиям (`1h`, `24h`, `7d`).
2. **Одинаковый шаг `step` в исходном PaySim CSV:** в CSV шаг равен 1 часу и содержит тысячи операций. Для исключения произвольных перестановок зафиксировано детерминированное правило вторичной сортировки: `(step ASC, source_row_idx ASC)` через устойчивый идентификатор `PAYSIM_{source_row_idx:08d}`.
3. **Одинаковое время (`t_1 == t_2`):** в офлайн-пайплайне сортировка по `(utc_timestamp, tx_id)`; в онлайн-скоринге — сериализация в порядке поступления под `RLock`.
4. **Идемпотентность и конфликты по `tx_id`:** повтор с идентичным каноническим снимком возвращает прежнее решение (`is_duplicate=True`) без повторного изменения окон; повтор с изменёнными параметрами отклоняется (`HTTP 409 Conflict`).
5. **Устаревшие / рассинхронизированные события:** отклоняются при отставании свыше `max_sender_out_of_order_seconds` или `max_global_staleness_seconds` (`StaleEventError` → `HTTP 422`).
