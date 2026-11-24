"""Generate a native, editable 16:9 light-themed PowerPoint (.pptx) presentation.

Builds a strictly ECMA-376 compliant OOXML (.pptx) archive using only Python's standard library
plus Pillow (for reading exact chart dimensions), embedding the 4 real light-themed project charts,
native editable shapes/cards, and a native PowerPoint table for the ablation study.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PRES_DIR = ROOT / "presentation"
ASSETS_DIR = PRES_DIR / "assets"
OUTPUT_PPTX = PRES_DIR / "fraud_agent_presentation.pptx"

# 16:9 Widescreen slide size in EMUs (13.333 in x 7.5 in)
SLIDE_W = 12192000
SLIDE_H = 6858000
EMU_PER_IN = 914400


def inch(val: float) -> int:
    return int(round(val * EMU_PER_IN))


def pt(val: float) -> int:
    return int(round(val * 100))


class SlideBuilder:
    """Helper to construct valid PresentationML slide XML with shapes, text, tables, and images."""

    def __init__(self, slide_num: int, total_slides: int, bg_hex: str = "F8FAFC"):
        self.slide_num = slide_num
        self.total_slides = total_slides
        self.bg_hex = bg_hex
        self._next_id = 2
        self._shapes: list[str] = []
        self._rels: list[tuple[str, str, str]] = [
            (
                "rId1",
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout",
                "../slideLayouts/slideLayout1.xml",
            )
        ]

    def _alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def add_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill_hex: str = "FFFFFF",
        line_hex: str | None = "CBD5E1",
        line_pt: float = 0.75,
        rounded: bool = True,
    ) -> None:
        sid = self._alloc_id()
        geom = "roundRect" if rounded else "rect"
        ln_xml = (
            f'<a:ln w="{int(line_pt * 12700)}"><a:solidFill><a:srgbClr val="{line_hex}"/></a:solidFill></a:ln>'
            if line_hex
            else '<a:ln><a:noFill/></a:ln>'
        )
        av_xml = '<a:avLst><a:gd name="adj" fmla="val 4000"/></a:avLst>' if rounded else "<a:avLst/>"
        self._shapes.append(
            f'<p:sp>'
            f'<p:nvSpPr><p:cNvPr id="{sid}" name="Rect {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>'
            f'<a:xfrm><a:off x="{inch(x)}" y="{inch(y)}"/><a:ext cx="{inch(w)}" cy="{inch(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="{geom}">{av_xml}</a:prstGeom>'
            f'<a:solidFill><a:srgbClr val="{fill_hex}"/></a:solidFill>'
            f'{ln_xml}'
            f'</p:spPr>'
            f'<p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>'
            f'</p:sp>'
        )

    def add_textbox(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        paragraphs: list[tuple[str, float, bool, str]],
        align: str = "l",
        line_spacing: int = 120000,
    ) -> None:
        """Add text box. Each paragraph is (text, font_pt, bold, color_hex)."""
        sid = self._alloc_id()
        p_xml_parts = []
        for text, fpt, bold, color in paragraphs:
            b_attr = ' b="1"' if bold else ' b="0"'
            safe_txt = escape(text)
            p_xml_parts.append(
                f'<a:p>'
                f'<a:pPr algn="{align}" spcAft="400"><a:lnSpc><a:spcPct val="{line_spacing}"/></a:lnSpc></a:pPr>'
                f'<a:r>'
                f'<a:rPr lang="ru-RU" sz="{pt(fpt)}"{b_attr} dirty="0">'
                f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
                f'<a:latin typeface="Segoe UI"/><a:ea typeface="Segoe UI"/><a:cs typeface="Segoe UI"/>'
                f'</a:rPr>'
                f'<a:t>{safe_txt}</a:t>'
                f'</a:r>'
                f'</a:p>'
            )
        self._shapes.append(
            f'<p:sp>'
            f'<p:nvSpPr><p:cNvPr id="{sid}" name="TextBox {sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>'
            f'<a:xfrm><a:off x="{inch(x)}" y="{inch(y)}"/><a:ext cx="{inch(w)}" cy="{inch(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            f'<a:noFill/><a:ln><a:noFill/></a:ln>'
            f'</p:spPr>'
            f'<p:txBody>'
            f'<a:bodyPr wrap="square" lIns="54000" tIns="36000" rIns="54000" bIns="36000" anchor="t"/>'
            f'<a:lstStyle/>'
            f'{"".join(p_xml_parts)}'
            f'</p:txBody>'
            f'</p:sp>'
        )

    def add_card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        lines: list[str],
        accent_hex: str = "0284C7",
        title_pt: float = 14.5,
        body_pt: float = 12.0,
    ) -> None:
        # Card background
        self.add_rect(x, y, w, h, fill_hex="FFFFFF", line_hex="CBD5E1", line_pt=0.75, rounded=True)
        # Left accent bar
        self.add_rect(x, y + 0.08, 0.07, h - 0.16, fill_hex=accent_hex, line_hex=None, rounded=False)
        paras: list[tuple[str, float, bool, str]] = [(title, title_pt, True, accent_hex)]
        for ln in lines:
            paras.append((ln, body_pt, False, "334155"))
        self.add_textbox(x + 0.16, y + 0.10, w - 0.24, h - 0.16, paras)

    def add_kpi_card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str,
        label: str,
        accent_hex: str = "0284C7",
    ) -> None:
        self.add_rect(x, y, w, h, fill_hex="FFFFFF", line_hex="CBD5E1", line_pt=0.75, rounded=True)
        self.add_rect(x, y + 0.08, 0.07, h - 0.16, fill_hex=accent_hex, line_hex=None, rounded=False)
        self.add_textbox(
            x + 0.18,
            y + 0.15,
            w - 0.24,
            h - 0.20,
            [
                (value, 26.0, True, "0F172A"),
                (label, 11.0, False, "475569"),
            ],
        )

    def add_header(self, tag: str, title: str) -> None:
        # Soft light pill tag
        self.add_rect(0.55, 0.35, 3.6, 0.32, fill_hex="E0F2FE", line_hex="7DD3FC", line_pt=0.6, rounded=True)
        self.add_textbox(0.62, 0.36, 3.45, 0.30, [(tag.upper(), 9.5, True, "0369A1")])
        self.add_textbox(0.55, 0.72, 12.2, 0.58, [(title, 24.0, True, "0F172A")])

    def add_footer(self) -> None:
        self.add_rect(0.55, 7.02, 12.23, 0.015, fill_hex="CBD5E1", line_hex=None, rounded=False)
        self.add_textbox(
            0.55, 7.07, 9.0, 0.32,
            [("AI Fraud Detection Agent — Многоуровневый ИИ-агент обнаружения мошеннических переводов", 9.5, False, "64748B")],
        )
        self.add_textbox(
            10.8, 7.07, 1.98, 0.32,
            [(f"{self.slide_num} / {self.total_slides}", 9.5, True, "0F172A")],
            align="r",
        )

    def add_image(self, img_rel_target: str, img_path: Path, x: float, y: float, max_w: float, max_h: float) -> None:
        # Preserve aspect ratio inside (max_w, max_h)
        with Image.open(img_path) as im:
            iw, ih = im.size
        scale = min(max_w / iw, max_h / ih)
        w = iw * scale
        h = ih * scale
        # Center inside box
        ox = x + (max_w - w) / 2.0
        oy = y + (max_h - h) / 2.0

        # Subtle white frame behind chart
        self.add_rect(x, y, max_w, max_h, fill_hex="FFFFFF", line_hex="CBD5E1", line_pt=0.75, rounded=True)

        rid = f"rId{len(self._rels) + 1}"
        self._rels.append((
            rid,
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            img_rel_target,
        ))
        sid = self._alloc_id()
        self._shapes.append(
            f'<p:pic>'
            f'<p:nvPicPr><p:cNvPr id="{sid}" name="Picture {sid}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr>'
            f'<a:xfrm><a:off x="{inch(ox)}" y="{inch(oy)}"/><a:ext cx="{inch(w)}" cy="{inch(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            f'</p:spPr>'
            f'</p:pic>'
        )

    def add_table(
        self,
        x: float,
        y: float,
        col_widths: list[float],
        rows: list[list[str]],
        highlight_last: bool = True,
    ) -> None:
        sid = self._alloc_id()
        total_w = sum(col_widths)
        row_h = 0.48
        total_h = row_h * len(rows)

        grid_cols = "".join(f'<a:gridCol w="{inch(cw)}"/>' for cw in col_widths)
        tr_xml = []
        for r_idx, row in enumerate(rows):
            is_header = r_idx == 0
            is_hl = highlight_last and r_idx == len(rows) - 1
            bg = "0F172A" if is_header else ("EDE9FE" if is_hl else ("F8FAFC" if r_idx % 2 == 0 else "FFFFFF"))
            fg = "FFFFFF" if is_header else ("4C1D95" if is_hl else "1E293B")
            bold = is_header or is_hl
            b_attr = ' b="1"' if bold else ' b="0"'
            cells = []
            for c_idx, val in enumerate(row):
                algn = "l" if c_idx == 0 else "r"
                cells.append(
                    f'<a:tc>'
                    f'<a:txBody><a:bodyPr anchor="ctr" lIns="72000" rIns="72000" tIns="36000" bIns="36000"/><a:lstStyle/>'
                    f'<a:p><a:pPr algn="{algn}"/><a:r>'
                    f'<a:rPr lang="ru-RU" sz="{pt(11.0 if is_header else 10.8)}"{b_attr} dirty="0">'
                    f'<a:solidFill><a:srgbClr val="{fg}"/></a:solidFill>'
                    f'<a:latin typeface="Segoe UI"/><a:ea typeface="Segoe UI"/><a:cs typeface="Segoe UI"/>'
                    f'</a:rPr><a:t>{escape(val)}</a:t></a:r></a:p></a:txBody>'
                    f'<a:tcPr anchor="ctr">'
                    f'<a:lnL w="6350"><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:lnL>'
                    f'<a:lnR w="6350"><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:lnR>'
                    f'<a:lnT w="6350"><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:lnT>'
                    f'<a:lnB w="6350"><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:lnB>'
                    f'<a:solidFill><a:srgbClr val="{bg}"/></a:solidFill>'
                    f'</a:tcPr>'
                    f'</a:tc>'
                )
            tr_xml.append(f'<a:tr h="{inch(row_h)}">{"".join(cells)}</a:tr>')

        self._shapes.append(
            f'<p:graphicFrame>'
            f'<p:nvGraphicFramePr><p:cNvPr id="{sid}" name="Table {sid}"/><p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr>'
            f'<p:xfrm><a:off x="{inch(x)}" y="{inch(y)}"/><a:ext cx="{inch(total_w)}" cy="{inch(total_h)}"/></p:xfrm>'
            f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">'
            f'<a:tbl><a:tblPr firstRow="1" bandRow="1"/><a:tblGrid>{grid_cols}</a:tblGrid>{"".join(tr_xml)}</a:tbl>'
            f'</a:graphicData></a:graphic>'
            f'</p:graphicFrame>'
        )

    def render_slide_xml(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            f'<p:cSld>'
            f'<p:bg><p:bgPr><a:solidFill><a:srgbClr val="{self.bg_hex}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>'
            f'<p:spTree>'
            f'<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            f'<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            f'{"".join(self._shapes)}'
            f'</p:spTree>'
            f'</p:cSld>'
            f'<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>'
            f'</p:sld>'
        )

    def render_rels_xml(self) -> str:
        items = "".join(
            f'<Relationship Id="{rid}" Type="{rtype}" Target="{target}"/>'
            for rid, rtype, target in self._rels
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{items}</Relationships>'
        )


def build_slides() -> list[SlideBuilder]:
    total = 11
    slides: list[SlideBuilder] = []

    # ------------------------------------------------------------------
    # SLIDE 1: TITLE
    # ------------------------------------------------------------------
    s1 = SlideBuilder(1, total, bg_hex="F8FAFC")
    s1.add_rect(0.0, 0.0, 13.333, 0.14, fill_hex="0284C7", line_hex=None, rounded=False)
    s1.add_rect(0.65, 0.85, 4.6, 0.36, fill_hex="E0F2FE", line_hex="7DD3FC", line_pt=0.6, rounded=True)
    s1.add_textbox(0.75, 0.88, 4.4, 0.30, [("FINTECH ANTI-FRAUD  ·  ИНЖЕНЕРНЫЙ ПРОЕКТ", 10.0, True, "0369A1")])
    s1.add_textbox(
        0.65, 1.45, 11.8, 1.55,
        [("ИИ-агент для обнаружения\nмошеннических переводов", 38.0, True, "0F172A")],
        line_spacing=112000,
    )
    s1.add_textbox(
        0.65, 3.15, 11.4, 1.0,
        [(
            "Многоуровневая система скоринга в реальном времени: Движок правил + ML (GBDT) + "
            "Детектор аномалий (Isolation Forest) + SHAP-объяснения + FastAPI + Streamlit + Мониторинг дрейфа (PSI)",
            16.0, False, "475569",
        )],
    )
    s1.add_kpi_card(0.65, 4.65, 2.85, 1.45, "0.9745", "PR-AUC ПОЛНОГО АГЕНТА", "0284C7")
    s1.add_kpi_card(3.72, 4.65, 2.85, 1.45, "99.31%", "СОХРАНЁННЫХ СРЕДСТВ", "16A34A")
    s1.add_kpi_card(6.79, 4.65, 2.85, 1.45, "35", "LEAK-FREE ПРИЗНАКОВ", "6D28D9")
    s1.add_kpi_card(9.86, 4.65, 2.85, 1.45, "4 уровня", "ALLOW · OTP · REVIEW · BLOCK", "D97706")
    s1.add_footer()
    slides.append(s1)

    # ------------------------------------------------------------------
    # SLIDE 2: PROBLEM
    # ------------------------------------------------------------------
    s2 = SlideBuilder(2, total)
    s2.add_header("01 · Постановка задачи", "Почему антифрод — это агентная система, а не одна модель")
    s2.add_card(
        0.55, 1.55, 5.95, 5.1,
        "Ограничения наивного подхода",
        [
            "• Сильный дисбаланс классов (~1.5% фрода): метрика Accuracy бессмысленна.",
            "• Асимметричная цена ошибки: пропуск стоит всей суммы перевода, ложная блокировка ведёт к оттоку клиента.",
            "• «Чёрный ящик»: аналитику недостаточно числа 0.87 — нужны конкретные причины и контекст.",
            "• Адаптивность мошенников: новые схемы быстро ломают статическую модель (дрейф концепции).",
        ],
        accent_hex="DC2626", title_pt=16.0, body_pt=13.5,
    )
    s2.add_card(
        6.82, 1.55, 5.95, 5.1,
        "Многоуровневый ИИ-агент",
        [
            "• Уровень 1 — Движок правил: мгновенные лимиты, чёрные списки и графовые эвристики.",
            "• Уровень 2 — ML + Аномалии: GBDT для известных паттернов + Isolation Forest для новых схем.",
            "• Уровень 3 — Решающий слой: выбор действия (ALLOW / OTP / REVIEW / BLOCK) по бизнес-стоимости + SHAP.",
            "• Цикл обратной связи: разметка аналитиков в Streamlit → дообучение + контроль дрейфа PSI.",
        ],
        accent_hex="16A34A", title_pt=16.0, body_pt=13.5,
    )
    s2.add_footer()
    slides.append(s2)

    # ------------------------------------------------------------------
    # SLIDE 3: ARCHITECTURE
    # ------------------------------------------------------------------
    s3 = SlideBuilder(3, total)
    s3.add_header("02 · Архитектура системы", "Сквозной пайплайн скоринга и обратной связи")
    steps = [
        ("1. Поток и валидация", "Имитация потока (replay.py) → FastAPI + Pydantic → Parquet / SQLite (PostgreSQL).", "0284C7"),
        ("2. Признаки (Leak-Free)", "OnlineFeatureStore: окна 1ч/24ч/7д, z-score клиента, новый получатель, граф (NetworkX).", "0284C7"),
        ("3. Скоринг (3 сигнала)", "Rule Engine + GBDT (балансировка классов) + Isolation Forest на легитимных операциях.", "6D28D9"),
        ("4. Решающий агент", "Композитный риск → ALLOW / OTP / REVIEW / BLOCK + SHAP-факторы и резюме кейса.", "16A34A"),
        ("5. Аналитик и MLOps", "Streamlit-дашборд, разметка (/feedback), мониторинг PSI (/drift) и дообучение (/retrain).", "D97706"),
    ]
    for i, (st_title, st_body, st_col) in enumerate(steps):
        sx = 0.55 + i * 2.48
        s3.add_rect(sx, 1.55, 2.32, 2.45, fill_hex="FFFFFF", line_hex="CBD5E1", line_pt=0.75, rounded=True)
        s3.add_rect(sx, 1.55, 2.32, 0.08, fill_hex=st_col, line_hex=None, rounded=False)
        s3.add_textbox(
            sx + 0.12, 1.72, 2.08, 2.15,
            [(st_title, 13.0, True, "0F172A"), (st_body, 11.2, False, "334155")],
        )
    s3.add_card(
        0.55, 4.35, 3.88, 2.25,
        "Единый код признаков",
        ["Идентичное вычисление признаков в обучении и в онлайн-API полностью исключает training-serving skew."],
        accent_hex="0284C7", title_pt=14.5, body_pt=12.5,
    )
    s3.add_card(
        4.72, 4.35, 3.88, 2.25,
        "Формула риска агента",
        ["risk = 0.65·ML + 0.25·Rules + 0.10·Anomaly (+ буст при аномалии > 0.85 и мгновенный блок по чёрным спискам)."],
        accent_hex="6D28D9", title_pt=14.5, body_pt=12.5,
    )
    s3.add_card(
        8.89, 4.35, 3.88, 2.25,
        "Готовность к деплою",
        ["Упаковано в Dockerfile и docker-compose.yml (API :8000 + Dashboard :8501), покрыто автотестами (7/7 OK)."],
        accent_hex="16A34A", title_pt=14.5, body_pt=12.5,
    )
    s3.add_footer()
    slides.append(s3)

    # ------------------------------------------------------------------
    # SLIDE 4: DATA & SIMULATOR
    # ------------------------------------------------------------------
    s4 = SlideBuilder(4, total)
    s4.add_header("03 · Данные и методология", "Симулятор (PaySim-формат) и строгое разбиение по времени")
    s4.add_image("../media/image1.png", ASSETS_DIR / "fraud_by_day.png", 0.55, 1.50, 7.25, 5.15)
    s4.add_card(
        8.05, 1.50, 4.72, 3.05,
        "30 000 транзакций · 30 дней · 1.50%",
        [
            "• account_takeover (захват аккаунта): 135",
            "• mule_chain (дропперы, общие устройства): 103",
            "• velocity_burst (серии за 1 час): 99",
            "• anomalous_amount (аномальная сумма): 67",
            "• novel_api_cashout (дрейф в Test): 46",
        ],
        accent_hex="0284C7", title_pt=14.0, body_pt=12.0,
    )
    s4.add_card(
        8.05, 4.75, 4.72, 1.90,
        "Реалистичность и честность оценки",
        [
            "Добавлено ~4% «сложных легитимных» переводов (крупные суммы, ночное время, обнуление счёта). Разбиение строго по времени: Train 60% / Val 20% / Test 20%.",
        ],
        accent_hex="D97706", title_pt=14.0, body_pt=11.8,
    )
    s4.add_footer()
    slides.append(s4)

    # ------------------------------------------------------------------
    # SLIDE 5: FEATURE ENGINEERING
    # ------------------------------------------------------------------
    s5 = SlideBuilder(5, total)
    s5.add_header("04 · Конструирование признаков", "35 признаков без утечки будущих данных (OnlineFeatureStore)")
    s5.add_card(
        0.55, 1.50, 5.95, 3.15,
        "Транзакционные (18 признаков)",
        [
            "• amount, log_amount, hour_of_day, is_night",
            "• balance_drain_ratio, orig_balance_zero_after",
            "• orig_balance_error, dest_balance_error, is_foreign_ip",
            "• One-hot: тип операции (TRANSFER, CASH_OUT, PAYMENT, CASH_IN, DEBIT) и канал (MOBILE, WEB, API, ATM)",
        ],
        accent_hex="0284C7", title_pt=15.0, body_pt=12.5,
    )
    s5.add_card(
        6.82, 1.50, 5.95, 3.15,
        "Поведенческие и отклонения (12 признаков)",
        [
            "• Скользящие окна по прошлым операциям: 1ч, 24ч, 7д (количество и сумма)",
            "• Отклонение от профиля клиента (алгоритм Welford): amount_zscore, amount_to_mean_ratio",
            "• Новизна: is_new_recipient, is_new_device, time_since_last_tx_hours",
        ],
        accent_hex="6D28D9", title_pt=15.0, body_pt=12.5,
    )
    s5.add_card(
        0.55, 4.90, 12.22, 1.75,
        "Графовые признаки (5 · инкрементальный расчёт + NetworkX)",
        [
            "dest_in_degree (уникальные отправители на получателя)  ·  orig_out_degree  ·  device_shared_accounts (общие устройства)  ·  dest_tx_count_24h  ·  is_mule_candidate.",
            "Отсутствие утечек из будущего подтверждено автоматическим юнит-тестом TestFeatureLeakage.",
        ],
        accent_hex="16A34A", title_pt=15.0, body_pt=12.5,
    )
    s5.add_footer()
    slides.append(s5)

    # ------------------------------------------------------------------
    # SLIDE 6: MODELING & DECISION POLICY
    # ------------------------------------------------------------------
    s6 = SlideBuilder(6, total)
    s6.add_header("05 · Моделирование и решающий слой", "Ансамбль сигналов и 4-уровневая политика действий")
    s6.add_card(
        0.55, 1.50, 3.88, 2.55,
        "Level 1: Rule Engine",
        ["Чёрные списки аккаунтов/устройств, лимит ≥ 250k, velocity ≥ 6/ч, опустошение счёта ≥ 95%, общие устройства (≥ 3), дропперы (≥ 5 отправителей)."],
        accent_hex="0284C7", title_pt=14.5, body_pt=12.2,
    )
    s6.add_card(
        4.72, 1.50, 3.88, 2.55,
        "Level 2: GBDT + IsoForest",
        ["GBDT (LightGBM / HistGBDT с class_weight='balanced') оценивает вероятность фрода. Isolation Forest обучается только на норме для ловли новых схем."],
        accent_hex="6D28D9", title_pt=14.5, body_pt=12.2,
    )
    s6.add_card(
        8.89, 1.50, 3.88, 2.55,
        "Level 3: Cost Thresholds",
        ["Пороги подобраны на Validation по бизнес-стоимости: пропущенный фрод = сумма перевода; OTP = 2; REVIEW = 25; ложный BLOCK = 150."],
        accent_hex="16A34A", title_pt=14.5, body_pt=12.2,
    )
    tiers = [
        ("ALLOW  (risk < 0.20)", "Мгновенное проведение без трения для клиента.", "16A34A"),
        ("OTP  (0.20 ≤ risk < 0.40)", "Доп. подтверждение (SMS/Push), низкая стоимость трения.", "0284C7"),
        ("REVIEW  (0.40 ≤ risk < 0.70)", "Удержание и передача кейса в очередь аналитика с SHAP.", "D97706"),
        ("BLOCK  (risk ≥ 0.70 / Rule)", "Автоматическая блокировка критически рискового перевода.", "DC2626"),
    ]
    for i, (t_title, t_body, t_col) in enumerate(tiers):
        s6.add_card(0.55 + i * 3.10, 4.35, 2.92, 2.25, t_title, [t_body], accent_hex=t_col, title_pt=13.2, body_pt=12.0)
    s6.add_footer()
    slides.append(s6)

    # ------------------------------------------------------------------
    # SLIDE 7: ABLATIONS VISUAL
    # ------------------------------------------------------------------
    s7 = SlideBuilder(7, total)
    s7.add_header("06 · Оценка качества и абляции", "Сравнение вариантов системы (Test Set, 6 000 транзакций)")
    s7.add_image("../media/image2.png", ASSETS_DIR / "ablation_comparison.png", 0.55, 1.42, 12.22, 3.65)
    s7.add_card(
        0.55, 5.22, 3.88, 1.52,
        "+0.145 PR-AUC от признаков",
        ["Поведенческие и графовые признаки подняли LogReg с 0.8135 до 0.9589 и снизили затраты FP для GBDT в 5 раз."],
        accent_hex="0284C7", title_pt=13.5, body_pt=11.2,
    )
    s7.add_card(
        4.72, 5.22, 3.88, 1.52,
        "Одни правила слабы",
        ["Rules Only дают PR-AUC 0.5622, Recall@1%FPR 51.8% и 498 ручных проверок вместо 13 у агента."],
        accent_hex="D97706", title_pt=13.5, body_pt=11.2,
    )
    s7.add_card(
        8.89, 5.22, 3.88, 1.52,
        "Полный ИИ-агент",
        ["PR-AUC 0.9745, Precision@100 0.99, чистая экономия 2.80 млн + защита от чёрных списков и аномалий."],
        accent_hex="6D28D9", title_pt=13.5, body_pt=11.2,
    )
    s7.add_footer()
    slides.append(s7)

    # ------------------------------------------------------------------
    # SLIDE 8: BUSINESS COST TABLE
    # ------------------------------------------------------------------
    s8 = SlideBuilder(8, total)
    s8.add_header("07 · Финансовый эффект", "Детальные метрики качества и бизнес-стоимости (Test Set)")
    s8.add_table(
        0.55, 1.55,
        [3.92, 1.25, 1.35, 1.15, 1.60, 1.35, 1.60],
        [
            ["Вариант системы", "PR-AUC", "Rec@1%FPR", "P@100", "Предотвращено", "Затраты FP", "Net Savings"],
            ["1. Rules Only", "0.5622", "0.5177", "0.7400", "2 642 393", "12 624", "2 629 769"],
            ["2. Baseline LogReg (Tx Only)", "0.8135", "0.7660", "0.9000", "1 875 759", "10 120", "1 865 639"],
            ["3. Baseline LogReg (All Features)", "0.9589", "0.9645", "0.9800", "2 788 031", "2 591", "2 785 440"],
            ["4. GBDT (Tx Only)", "0.9660", "0.9504", "1.0000", "2 817 788", "6 619", "2 811 169"],
            ["5. GBDT (Full Features)", "0.9876", "0.9929", "1.0000", "2 804 927", "1 329", "2 803 598"],
            ["6. Isolation Forest Only", "0.5199", "0.0000", "0.6000", "2 810 960", "30 553", "2 780 407"],
            ["7. Full AI Agent (Rules + GBDT + IsoForest)", "0.9745", "0.9716", "0.9900", "2 806 239", "1 543", "2 804 696"],
        ],
        highlight_last=True,
    )
    s8.add_card(
        0.55, 5.65, 12.22, 1.05,
        "Итог по бизнес-стоимости",
        ["Общий объём фрода на тестовой выборке: 2 824 069. Полный ИИ-агент сохраняет 99.31% суммы при всего 13 ручных проверках (REVIEW) и 137 блокировках (BLOCK)."],
        accent_hex="16A34A", title_pt=13.5, body_pt=12.0,
    )
    s8.add_footer()
    slides.append(s8)

    # ------------------------------------------------------------------
    # SLIDE 9: EXPLAINABILITY & DASHBOARD
    # ------------------------------------------------------------------
    s9 = SlideBuilder(9, total)
    s9.add_header("08 · Объяснимость и интерфейс аналитика", "SHAP-разбор кейса, LLM-резюме и дашборд Streamlit")
    s9.add_image("../media/image3.png", ASSETS_DIR / "shap_case_example.png", 0.55, 1.50, 7.15, 5.15)
    s9.add_card(
        7.95, 1.50, 4.82, 2.45,
        "Объяснение для каждого алерта",
        [
            "• Топ-5 факторов SHAP переводятся на понятный язык («сумма в 12× выше средней», «счёт обнулён», «новый получатель»).",
            "• Опциональный LLM-слой пишет краткое резюме строго поверх SHAP и не участвует в скоринге.",
        ],
        accent_hex="6D28D9", title_pt=14.0, body_pt=11.8,
    )
    s9.add_card(
        7.95, 4.15, 4.82, 2.50,
        "4 вкладки Streamlit-дашборда",
        [
            "• Очередь алертов и карточка кейса + форма разметки.",
            "• Мониторинг PSI и кнопка дообучения модели.",
            "• Абляции и бизнес-стоимость.",
            "• Интерактивный симулятор ручной проверки перевода.",
        ],
        accent_hex="0284C7", title_pt=14.0, body_pt=11.8,
    )
    s9.add_footer()
    slides.append(s9)

    # ------------------------------------------------------------------
    # SLIDE 10: DRIFT & FEEDBACK LOOP
    # ------------------------------------------------------------------
    s10 = SlideBuilder(10, total)
    s10.add_header("09 · Мониторинг и MLOps", "Детекция дрейфа (PSI) и цикл обратной связи")
    s10.add_image("../media/image4.png", ASSETS_DIR / "psi_drift.png", 0.55, 1.50, 7.15, 5.15)
    s10.add_card(
        7.95, 1.50, 4.82, 2.45,
        "Статус дрейфа: CRITICAL",
        [
            "PSI автоматически зафиксировал сдвиг в позднем периоде:",
            "• dest_in_degree: PSI = 2.003 (рост графа и дроппер-счетов).",
            "• risk_score: PSI = 0.741 (появление новой схемы novel_api_cashout).",
        ],
        accent_hex="DC2626", title_pt=14.0, body_pt=12.0,
    )
    s10.add_card(
        7.95, 4.15, 4.82, 2.50,
        "Цикл обратной связи (Feedback Loop)",
        [
            "Аналитик размечает кейс в дашборде → POST /feedback → сохранение в БД → POST /retrain дообучает GBDT и Isolation Forest с весом свежих меток 3× и пересчитывает пороги.",
        ],
        accent_hex="16A34A", title_pt=14.0, body_pt=12.0,
    )
    s10.add_footer()
    slides.append(s10)

    # ------------------------------------------------------------------
    # SLIDE 11: RISKS & CHECKLIST
    # ------------------------------------------------------------------
    s11 = SlideBuilder(11, total)
    s11.add_header("10 · Риски, ограничения и итоги", "Риски внедрения и результаты проекта")
    s11.add_card(
        0.55, 1.50, 5.95, 5.15,
        "Ключевые риски и меры снижения",
        [
            "• Дрейф концепции → PSI-мониторинг + Isolation Forest + регулярное дообучение.",
            "• Запаздывающие метки → учёт лага подтверждения chargeback при формировании окна обучения.",
            "• Синтетические данные → метрики симулятора не переносятся напрямую в банк без пилота.",
            "• Ложные срабатывания → мягкие действия OTP и REVIEW вместо жёсткого бинарного блока.",
            "• Приватность и справедливость → псевдонимизация ID, отказ от защищённых атрибутов и PII.",
        ],
        accent_hex="D97706", title_pt=15.0, body_pt=12.8,
    )
    s11.add_card(
        6.82, 1.50, 5.95, 5.15,
        "Чек-лист результатов (выполнено 100%)",
        [
            "• Воспроизводимый репозиторий с README.md и config.yaml.",
            "• Ноутбуки EDA и абляций (01_eda, 02_modeling).",
            "• FastAPI-сервис, поток replay.py, Dockerfile и docker-compose.yml.",
            "• Рабочий Streamlit-дашборд аналитика с предзаполненной БД.",
            "• Отчёт project_report.md, PDF-руководство user_guide.pdf и 7/7 пройденных автотестов.",
        ],
        accent_hex="16A34A", title_pt=15.0, body_pt=12.8,
    )
    s11.add_footer()
    slides.append(s11)

    return slides


def write_pptx() -> Path:
    slides = build_slides()
    n = len(slides)

    media_files = [
        ("image1.png", ASSETS_DIR / "fraud_by_day.png"),
        ("image2.png", ASSETS_DIR / "ablation_comparison.png"),
        ("image3.png", ASSETS_DIR / "shap_case_example.png"),
        ("image4.png", ASSETS_DIR / "psi_drift.png"),
    ]

    slide_overrides = "".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, n + 1)
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        f'{slide_overrides}'
        '</Types>'
    )

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        '</Relationships>'
    )

    sld_id_list = "".join(
        f'<p:sldId id="{255 + i}" r:id="rId{i + 1}"/>'
        for i in range(1, n + 1)
    )
    presentation_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        f'<p:sldIdLst>{sld_id_list}</p:sldIdLst>'
        f'<p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="screen16x9"/>'
        '<p:notesSz cx="6858000" cy="9144000"/>'
        '</p:presentation>'
    )

    pres_rels_items = (
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
        + "".join(
            f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
            for i in range(1, n + 1)
        )
    )
    presentation_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{pres_rels_items}</Relationships>'
    )

    slide_master_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="F8FAFC"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>'
        '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '</p:spTree></p:cSld>'
        '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
        '</p:sldMaster>'
    )

    slide_master_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>'
        '</Relationships>'
    )

    slide_layout_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">'
        '<p:cSld name="Blank"><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
    )

    slide_layout_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
        '</Relationships>'
    )

    theme_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="LightFinTech">'
        '<a:themeElements>'
        '<a:clrScheme name="LightFinTech">'
        '<a:dk1><a:srgbClr val="0F172A"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>'
        '<a:dk2><a:srgbClr val="334155"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2>'
        '<a:accent1><a:srgbClr val="0284C7"/></a:accent1><a:accent2><a:srgbClr val="6D28D9"/></a:accent2>'
        '<a:accent3><a:srgbClr val="16A34A"/></a:accent3><a:accent4><a:srgbClr val="D97706"/></a:accent4>'
        '<a:accent5><a:srgbClr val="DC2626"/></a:accent5><a:accent6><a:srgbClr val="0369A1"/></a:accent6>'
        '<a:hlink><a:srgbClr val="0284C7"/></a:hlink><a:folHlink><a:srgbClr val="6D28D9"/></a:folHlink>'
        '</a:clrScheme>'
        '<a:fontScheme name="SegoeUI">'
        '<a:majorFont><a:latin typeface="Segoe UI"/><a:ea typeface="Segoe UI"/><a:cs typeface="Segoe UI"/></a:majorFont>'
        '<a:minorFont><a:latin typeface="Segoe UI"/><a:ea typeface="Segoe UI"/><a:cs typeface="Segoe UI"/></a:minorFont>'
        '</a:fontScheme>'
        '<a:fmtScheme name="Office">'
        '<a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>'
        '<a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>'
        '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>'
        '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>'
        '</a:fmtScheme>'
        '</a:themeElements></a:theme>'
    )

    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/">'
        '<dc:title>ИИ-агент для обнаружения мошеннических переводов</dc:title>'
        '<dc:creator>AI Fraud Detection Agent</dc:creator>'
        '</cp:coreProperties>'
    )

    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
        '<Application>Microsoft Office PowerPoint</Application>'
        f'<Slides>{n}</Slides>'
        '</Properties>'
    )

    with zipfile.ZipFile(OUTPUT_PPTX, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("docProps/app.xml", app_xml)
        zf.writestr("ppt/presentation.xml", presentation_xml)
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels)
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml)
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels)
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml)
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels)
        zf.writestr("ppt/theme/theme1.xml", theme_xml)

        for img_name, img_path in media_files:
            zf.write(img_path, f"ppt/media/{img_name}")

        for i, slide in enumerate(slides, start=1):
            zf.writestr(f"ppt/slides/slide{i}.xml", slide.render_slide_xml())
            zf.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide.render_rels_xml())

    print(f"Generated {OUTPUT_PPTX} ({n} slides, {OUTPUT_PPTX.stat().st_size:,} bytes)")
    return OUTPUT_PPTX


if __name__ == "__main__":
    write_pptx()
