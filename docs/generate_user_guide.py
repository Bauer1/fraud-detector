"""Generate a multi-page A4 PDF User Guide for the AI Fraud Detection Agent web interface."""
from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = ROOT / "presentation" / "assets"
OUTPUT_PDF = DOCS_DIR / "user_guide.pdf"


def build_html() -> str:
    shap_img = "presentation/assets/shap_case_example.png"
    psi_img = "presentation/assets/psi_drift.png"
    ablation_img = "presentation/assets/ablation_comparison.png"

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<style>
  @font-face {{
    font-family: "MainFont";
    src: url("file:///C:/Windows/Fonts/arial.ttf");
    font-weight: normal;
  }}
  @font-face {{
    font-family: "MainFont";
    src: url("file:///C:/Windows/Fonts/arialbd.ttf");
    font-weight: bold;
  }}
  @font-face {{
    font-family: "MonoFont";
    src: url("file:///C:/Windows/Fonts/consola.ttf");
  }}
  body {{
    font-family: "MainFont", sans-serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #1e293b;
  }}
  .cover {{
    background-color: #0f172a;
    color: #f8fafc;
    padding: 26px 28px;
    border-radius: 8px;
    margin-bottom: 18px;
  }}
  .cover-tag {{
    font-size: 8.5pt;
    color: #38bdf8;
    font-weight: bold;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
  }}
  .cover h1 {{
    font-size: 20pt;
    color: #ffffff;
    margin: 0 0 8px 0;
  }}
  .cover p {{
    font-size: 10pt;
    color: #cbd5e1;
    margin: 0;
  }}
  h2 {{
    font-size: 13.5pt;
    color: #0f172a;
    border-bottom: 2px solid #0284c7;
    padding-bottom: 4px;
    margin-top: 18px;
    margin-bottom: 10px;
  }}
  h3 {{
    font-size: 11pt;
    color: #0369a1;
    margin-top: 12px;
    margin-bottom: 6px;
  }}
  p {{ margin: 0 0 8px 0; }}
  ul, ol {{ margin: 4px 0 10px 0; padding-left: 18px; }}
  li {{ margin-bottom: 4px; }}
  code {{
    font-family: "MonoFont", monospace;
    font-size: 8.8pt;
    background-color: #f1f5f9;
    color: #0f172a;
    padding: 1px 4px;
  }}
  pre {{
    font-family: "MonoFont", monospace;
    font-size: 8.5pt;
    background-color: #0f172a;
    color: #e2e8f0;
    padding: 10px 12px;
    margin: 8px 0 12px 0;
  }}
  .callout {{
    background-color: #eff6ff;
    border-left: 4px solid #0284c7;
    padding: 9px 12px;
    margin: 10px 0 12px 0;
  }}
  .warn {{
    background-color: #fffbeb;
    border-left: 4px solid #d97706;
    padding: 9px 12px;
    margin: 10px 0 12px 0;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0 12px 0;
    font-size: 9pt;
  }}
  th {{
    background-color: #0f172a;
    color: #ffffff;
    text-align: left;
    padding: 6px 8px;
    font-weight: bold;
  }}
  td {{
    border: 1px solid #cbd5e1;
    padding: 5px 8px;
    vertical-align: top;
  }}
  tr:nth-child(even) td {{
    background-color: #f8fafc;
  }}
  .img-wrap {{
    text-align: center;
    margin: 10px 0 12px 0;
  }}
  .img-wrap img {{
    width: 92%;
  }}
  .caption {{
    font-size: 8.5pt;
    color: #64748b;
    margin-top: 4px;
  }}
  .badge-allow {{ color: #15803d; font-weight: bold; }}
  .badge-otp {{ color: #0369a1; font-weight: bold; }}
  .badge-review {{ color: #b45309; font-weight: bold; }}
  .badge-block {{ color: #b91c1c; font-weight: bold; }}
</style>
</head>
<body>

<div class="cover">
  <div class="cover-tag">РУКОВОДСТВО ПОЛЬЗОВАТЕЛЯ ВЕБ-ИНТЕРФЕЙСА · ВЕРСИЯ 1.0</div>
  <h1>ИИ-агент обнаружения мошеннических переводов</h1>
  <p>Практическое руководство по работе с дашбордом аналитика (Streamlit) и REST API (FastAPI Swagger UI): обработка очереди алертов, разбор SHAP-объяснений, разметка кейсов, мониторинг дрейфа PSI и дообучение модели.</p>
</div>

<h2>1. Назначение системы и целевая аудитория</h2>
<p>Веб-система <b>AI Fraud Detection Agent</b> предназначена для оперативного мониторинга денежных переводов, автоматической оценки риска мошенничества и поддержки принятия решений антифрод-аналитиками.</p>
<ul>
  <li><b>Антифрод-аналитики</b> — разбирают очередь подозрительных переводов, изучают факторы риска (правила + SHAP), подтверждают или отклоняют алерты.</li>
  <li><b>Риск-менеджеры</b> — отслеживают бизнес-эффект (предотвращённый фрод, затраты на ложные срабатывания) и результаты абляций.</li>
  <li><b>ML-инженеры и сопровождение</b> — контролируют дрейф данных (PSI), запускают цикл дообучения по разметке и интегрируют сервис через FastAPI.</li>
</ul>

<div class="callout">
  <b>Как система принимает решение:</b> каждый перевод проходит 3 уровня проверки — (1) Движок правил, (2) ML-модель (GBDT) + Детектор аномалий (Isolation Forest), (3) Решающий агент, который вычисляет итоговый риск от 0 до 1 и назначает действие:
  <span class="badge-allow">ALLOW</span>, <span class="badge-otp">OTP</span>, <span class="badge-review">REVIEW</span> или <span class="badge-block">BLOCK</span>.
</div>

<h2>2. Запуск и вход в веб-интерфейс</h2>
<h3>Вариант А: Локальный запуск (Python)</h3>
<pre>1. uvicorn src.api.app:app --reload --port 8000
2. streamlit run dashboard/app.py
3. python -m src.stream.replay --n-events 300 --delay-ms 10  # имитация потока</pre>

<h3>Вариант Б: Запуск через Docker Compose</h3>
<pre>docker compose up --build</pre>

<table>
  <tr>
    <th>Компонент</th>
    <th>Адрес (URL)</th>
    <th>Назначение</th>
  </tr>
  <tr>
    <td><b>Дашборд аналитика (Streamlit)</b></td>
    <td><code>http://localhost:8501</code></td>
    <td>Основной веб-сайт для работы с алертами, кейсами, PSI и тестами</td>
  </tr>
  <tr>
    <td><b>Документация API (Swagger UI)</b></td>
    <td><code>http://localhost:8000/docs</code></td>
    <td>Интерактивный вызов REST API скоринга, разметки и дообучения</td>
  </tr>
</table>

<h3>Боковая панель состояния (Sidebar)</h3>
<p>В левой части сайта постоянно отображается панель состояния системы:</p>
<ul>
  <li><b>Статус подключения FastAPI</b>: зелёная плашка означает работу через сервер API; жёлтая плашка («локальный режим») позволяет полноценно работать напрямую с локальной БД и сохранёнными моделями даже при выключенном API.</li>
  <li><b>Активная модель и пороги агента</b>: отображаются актуальные оптимизированные пороги (по умолчанию <code>OTP &ge; 0.200</code>, <code>REVIEW &ge; 0.400</code>, <code>BLOCK &ge; 0.700</code>).</li>
  <li>Кнопка <b>«Обновить данные»</b> принудительно перезагружает статистику и очередь из базы данных.</li>
</ul>

<h2>3. Вкладка 1: «Очередь алертов и кейсы»</h2>
<p>Главная рабочая область антифрод-аналитика. В верхней части отображаются 5 ключевых KPI: <b>Всего проверено</b>, <b>Доля алертов</b>, <b>Заблокировано (сумма)</b>, <b>Размечено аналитиками</b> и <b>Средняя задержка (мс)</b>.</p>

<h3>3.1. Фильтрация очереди</h3>
<ul>
  <li><b>Фильтр по действию агента</b> — мультиселект действий (<code>BLOCK</code>, <code>REVIEW</code>, <code>OTP</code>, <code>ALLOW</code>). По умолчанию скрыты обычные операции <code>ALLOW</code>, чтобы аналитик видел только алерты.</li>
  <li><b>«Только неразмеченные кейсы»</b> — оставляет в таблице исключительно транзакции, по которым ещё не вынесен вердикт.</li>
  <li>Таблица автоматически отсортирована по убыванию итогового риска (<code>risk_score</code>) — самые критичные инциденты всегда наверху.</li>
</ul>

<h3>3.2. Работа с карточкой кейса</h3>
<p>Выберите нужный <code>tx_id</code> в выпадающем списке <b>«Выберите транзакцию для разбора»</b>. Карточка разделена на две колонки:</p>
<ul>
  <li><b>Левая колонка</b>: итоговое решение агента, текстовое резюме (SHAP / LLM), раскладка скоров (<code>Итоговый риск</code>, <code>ML-скор</code>, <code>Правила</code>, <code>Аномалия</code>), JSON с деталями перевода (отправитель, получатель, устройство, канал, страна IP) и список сработавших правил Уровня 1.</li>
  <li><b>Правая колонка</b>: интерактивная диаграмма <b>SHAP</b> (красные столбцы — факторы, повысившие риск; зелёные — снизившие риск) с пояснениями на русском языке, а также форма разметки аналитика.</li>
</ul>

<div class="img-wrap">
  <img src="{shap_img}" alt="SHAP пример">
  <div class="caption">Рис. 1. Пример графика локального SHAP-объяснения в карточке кейса.</div>
</div>

<h3>3.3. Пошаговая разметка кейса (Обратная связь)</h3>
<ol>
  <li>Изучите сработавшие правила, график SHAP и параметры перевода.</li>
  <li>В блоке <b>«Разметка аналитика»</b> выберите вердикт: <b>Мошенничество (Fraud)</b> или <b>Легитимный перевод (Legit)</b>.</li>
  <li>Укажите свой <b>ID аналитика</b> и при необходимости добавьте краткий комментарий по расследованию.</li>
  <li>Нажмите <b>«Сохранить разметку»</b> — вердикт запишется в БД и будет автоматически учтён при следующем дообучении модели.</li>
</ol>

<h2>4. Вкладка 2: «Мониторинг и дрейф (PSI)»</h2>
<p>Раздел предназначен для контроля деградации модели и выявления новых мошеннических схем (Concept Drift) с помощью индекса стабильности популяции <b>PSI (Population Stability Index)</b>.</p>

<table>
  <tr>
    <th>Значение PSI</th>
    <th>Статус</th>
    <th>Рекомендуемое действие</th>
  </tr>
  <tr>
    <td><code>PSI &lt; 0.10</code></td>
    <td><span class="badge-allow">OK</span></td>
    <td>Распределение стабильно, вмешательство не требуется.</td>
  </tr>
  <tr>
    <td><code>0.10 &le; PSI &lt; 0.25</code></td>
    <td><span class="badge-review">WARNING</span></td>
    <td>Умеренный сдвиг данных — усилить выборочный аудит алертов.</td>
  </tr>
  <tr>
    <td><code>PSI &ge; 0.25</code></td>
    <td><span class="badge-block">CRITICAL</span></td>
    <td>Сильный дрейф — накопить свежую разметку и запустить дообучение.</td>
  </tr>
</table>

<div class="img-wrap">
  <img src="{psi_img}" alt="PSI мониторинг">
  <div class="caption">Рис. 2. Диаграмма PSI по ключевым признакам и скорам модели.</div>
</div>

<h3>Запуск цикла дообучения</h3>
<p>Нажмите кнопку <b>«Запустить дообучение модели с учётом разметки аналитиков»</b> в нижней части вкладки. Система объединит исторические данные со свежей разметкой (с повышенным весом 3x), переобучит GBDT и Isolation Forest, заново оптимизирует пороги по бизнес-стоимости и выведет обновлённые метрики на тестовой выборке.</p>

<h2>5. Вкладка 3: «Абляции и бизнес-стоимость»</h2>
<p>Вкладка позволяет риск-менеджерам оценить вклад каждого компонента системы (правила, базовая модель, поведенческие/графовые признаки, детектор аномалий и полный агент) по метрикам <b>PR-AUC</b>, <b>ROC-AUC</b>, <b>Recall @ 1% FPR</b>, <b>Precision@100</b> и <b>Чистой экономии (Net Savings)</b>.</p>

<div class="img-wrap">
  <img src="{ablation_img}" alt="Абляции и экономия">
  <div class="caption">Рис. 3. Сравнение вариантов системы по PR-AUC и чистой финансовой экономии.</div>
</div>

<h2>6. Вкладка 4: «Тестовый скоринг транзакции»</h2>
<p>Интерактивная песочница позволяет вручную смоделировать любой перевод и мгновенно увидеть реакцию ИИ-агента, задержку расчёта, сработавшие правила и SHAP-объяснение. Проверенная транзакция сразу попадает в общую очередь алертов.</p>

<h3>Рекомендуемые тестовые сценарии</h3>
<table>
  <tr>
    <th>Сценарий</th>
    <th>Ключевые параметры формы</th>
    <th>Ожидаемое решение</th>
  </tr>
  <tr>
    <td><b>1. Чёрный список</b></td>
    <td>Отправитель: <code>C_BLACKLIST_001</code> или устройство <code>DEV_BLACK_001</code></td>
    <td><span class="badge-block">BLOCK</span> (риск 1.00, правило HARD_BLOCK)</td>
  </tr>
  <tr>
    <td><b>2. Захват аккаунта ночью</b></td>
    <td>Тип <code>TRANSFER</code>, сумма <code>75000</code>, баланс ДО <code>76000</code>, ПОСЛЕ <code>1000</code>, час <code>3</code>, получатель <code>C_MULE_001</code>, IP <code>NL</code></td>
    <td><span class="badge-block">BLOCK</span> или <span class="badge-review">REVIEW</span> (высокий ML + правила)</td>
  </tr>
  <tr>
    <td><b>3. Обычный платёж</b></td>
    <td>Тип <code>PAYMENT</code>, сумма <code>450</code>, баланс ДО <code>25000</code>, ПОСЛЕ <code>24550</code>, час <code>14</code>, канал <code>MOBILE</code>, IP <code>RU</code></td>
    <td><span class="badge-allow">ALLOW</span> (низкий риск)</td>
  </tr>
</table>

<h2>7. Справочник REST API (Swagger UI)</h2>
<p>Все функции веб-сайта также доступны программно по адресу <code>http://localhost:8000/docs</code>:</p>
<table>
  <tr>
    <th>Метод и путь</th>
    <th>Описание</th>
  </tr>
  <tr><td><code>GET /health</code></td><td>Проверка готовности сервиса, активной модели и текущих порогов.</td></tr>
  <tr><td><code>POST /score</code></td><td>Скоринг одной транзакции в реальном времени (возвращает действие, скоры, правила и SHAP).</td></tr>
  <tr><td><code>GET /alerts</code></td><td>Получение очереди алертов с фильтрами <code>action</code>, <code>only_unlabeled</code>, <code>limit</code>.</td></tr>
  <tr><td><code>GET /alerts/{{tx_id}}</code></td><td>Детальная карточка конкретной транзакции.</td></tr>
  <tr><td><code>POST /feedback</code></td><td>Сохранение вердикта аналитика (<code>is_fraud</code>, <code>analyst_id</code>, <code>notes</code>).</td></tr>
  <tr><td><code>GET /metrics/summary</code></td><td>Операционные KPI и Precision/Recall по размеченным кейсам.</td></tr>
  <tr><td><code>GET /monitoring/drift</code></td><td>Отчёт PSI по дрейфу живого потока относительно обучающей выборки.</td></tr>
  <tr><td><code>POST /retrain</code></td><td>Запуск переобучения моделей с учётом разметки аналитиков.</td></tr>
</table>

<h2>8. Частые вопросы и устранение неполадок</h2>
<div class="warn">
  <b>Очередь алертов пуста:</b> запустите имитацию потока командой <code>python -m src.stream.replay --in-process --n-events 150 --delay-ms 0</code> (или отправьте перевод через вкладку «Тестовый скоринг транзакции») и нажмите «Обновить данные».
</div>
<ul>
  <li><b>Как изменить лимиты, веса или чёрные списки?</b> Отредактируйте файл <code>config/config.yaml</code> (секции <code>rules</code> и <code>agent</code>) и перезапустите сервис.</li>
  <li><b>Как включить генерацию текстовых резюме через LLM?</b> Задайте переменную окружения <code>OPENAI_API_KEY</code> и установите <code>llm.enabled: true</code> в <code>config/config.yaml</code>. LLM используется строго для формулировки резюме поверх SHAP и не влияет на скоринг.</li>
  <li><b>Предупреждение «FastAPI не запущен — локальный режим»:</b> для работы по сети запустите <code>uvicorn src.api.app:app --port 8000</code> в отдельном терминале и обновите страницу дашборда.</li>
</ul>

</body>
</html>
"""


def generate_pdf() -> Path:
    html = build_html()
    archive = pymupdf.Archive(str(ROOT))
    story = pymupdf.Story(html=html, archive=archive)

    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (42, 42, -42, -48)  # margins: left, top, right, bottom

    base_pdf = DOCS_DIR / "_base_story.pdf"
    writer = pymupdf.DocumentWriter(str(base_pdf))
    more = 1
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()

    # Stamp clean Cyrillic page numbers in footer using Arial TTF and save final PDF once
    doc = pymupdf.open(str(base_pdf))
    total = len(doc)
    arial_path = "C:/Windows/Fonts/arial.ttf"
    for i, page in enumerate(doc, start=1):
        page.insert_font(fontname="ArialRU", fontfile=arial_path)
        footer_rect = pymupdf.Rect(42, mediabox.height - 30, mediabox.width - 42, mediabox.height - 14)
        page.insert_textbox(
            footer_rect,
            f"AI Fraud Detection Agent — Руководство пользователя   |   Страница {i} из {total}",
            fontsize=8,
            fontname="ArialRU",
            color=(0.39, 0.45, 0.55),
            align=pymupdf.TEXT_ALIGN_RIGHT,
        )
    doc.save(str(OUTPUT_PDF), garbage=4, deflate=True)
    doc.close()
    for leftover in (base_pdf, DOCS_DIR / "user_guide_tmp.pdf"):
        if leftover.exists():
            try:
                leftover.unlink()
            except OSError:
                pass

    print(f"Generated {OUTPUT_PDF} ({total} pages, {OUTPUT_PDF.stat().st_size:,} bytes)")
    return OUTPUT_PDF


if __name__ == "__main__":
    generate_pdf()
