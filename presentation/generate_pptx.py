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
    total = 6
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
            "Многоуровневая система скоринга в реальном времени: Правила + GBDT + "
            "Isolation Forest + SHAP + FastAPI + Streamlit + PSI-мониторинг",
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
    # SLIDE 2: PROBLEM & ARCHITECTURE
    # ------------------------------------------------------------------
    s2 = SlideBuilder(2, total)
    s2.add_header("01 · Постановка задачи и архитектура", "Почему антифрод — это агентный пайплайн, а не одна модель")
    s2.add_card(
        0.55, 1.48, 5.95, 2.35,
        "Почему одной ML-модели мало",
        [
            "• Дисбаланс ~1.5% фрода (Accuracy бессмысленна) и асимметричная цена ошибки.",
            "• Нужна объяснимость для аналитиков и защита от новых схем (дрейф концепции).",
        ],
        accent_hex="DC2626", title_pt=14.5, body_pt=12.2,
    )
    s2.add_card(
        6.82, 1.48, 5.95, 2.35,
        "Ключевые инженерные гарантии",
        [
            "• Единый OnlineFeatureStore в обучении и продакшене (нет training-serving skew).",
            "• Риск: 0.65·ML + 0.25·Rules + 0.10·Anomaly (+ буст > 0.85, блок по чёрным спискам).",
        ],
        accent_hex="16A34A", title_pt=14.5, body_pt=12.2,
    )
    steps = [
        ("1. Поток и API", "replay.py → FastAPI + Pydantic → SQLite / Parquet.", "0284C7"),
        ("2. Признаки", "OnlineFeatureStore: 35 признаков без утечек.", "0284C7"),
        ("3. Скоринг", "Rule Engine + GBDT + Isolation Forest.", "6D28D9"),
        ("4. Решение", "ALLOW / OTP / REVIEW / BLOCK + SHAP.", "16A34A"),
        ("5. MLOps", "Streamlit, /feedback, PSI и /retrain.", "D97706"),
    ]
    for i, (st_title, st_body, st_col) in enumerate(steps):
        sx = 0.55 + i * 2.48
        s2.add_rect(sx, 4.10, 2.32, 2.45, fill_hex="FFFFFF", line_hex="CBD5E1", line_pt=0.75, rounded=True)
        s2.add_rect(sx, 4.10, 2.32, 0.08, fill_hex=st_col, line_hex=None, rounded=False)
        s2.add_textbox(
            sx + 0.12, 4.28, 2.08, 2.15,
            [(st_title, 13.5, True, "0F172A"), (st_body, 11.8, False, "334155")],
        )
    s2.add_footer()
    slides.append(s2)

    # ------------------------------------------------------------------
    # SLIDE 3: DATA & LEAK-FREE FEATURES
    # ------------------------------------------------------------------
    s3 = SlideBuilder(3, total)
    s3.add_header("02 · Данные и признаки (Leak-Free)", "Симулятор PaySim (30 000 tx) и 35 признаков без утечек")
    s3.add_image("../media/image1.png", ASSETS_DIR / "fraud_by_day.png", 0.55, 1.48, 6.55, 5.15)
    s3.add_card(
        7.35, 1.48, 5.42, 2.45,
        "Симулятор и разбиение по времени",
        [
            "• 30 000 tx за 30 дней, фрод 1.50% (450 кейсов, 5 сценариев + дрейф novel_api_cashout в Test).",
            "• ~4% «сложных легитимных» переводов + строгий time split: Train 60% / Val 20% / Test 20%.",
        ],
        accent_hex="0284C7", title_pt=14.0, body_pt=11.8,
    )
    s3.add_card(
        7.35, 4.12, 5.42, 2.51,
        "35 признаков (тест TestFeatureLeakage)",
        [
            "• Транзакционные (18): сумма, час/ночь, drain ratio, ошибки баланса, IP, тип, канал.",
            "• Поведенческие (12): окна 1ч/24ч/7д, z-score (Welford), новизна получателя/устройства.",
            "• Графовые (5): dest_in_degree, orig_out_degree, общие устройства, is_mule_candidate.",
        ],
        accent_hex="6D28D9", title_pt=14.0, body_pt=11.8,
    )
    s3.add_footer()
    slides.append(s3)

    # ------------------------------------------------------------------
    # SLIDE 4: ABLATIONS, POLICY & BUSINESS IMPACT
    # ------------------------------------------------------------------
    s4 = SlideBuilder(4, total)
    s4.add_header("03 · Качество, политика действий и бизнес-эффект", "Абляции на Test Set (6 000 tx) и 4-уровневый отклик")
    s4.add_image("../media/image2.png", ASSETS_DIR / "ablation_comparison.png", 0.55, 1.45, 12.22, 3.25)
    s4.add_card(
        0.55, 4.88, 5.95, 1.80,
        "4 уровня действий (пороги по стоимости на Val)",
        [
            "• ALLOW (< 0.20)  ·  OTP (0.20–0.40, цена 2)  ·  REVIEW (0.40–0.70, цена 25)  ·  BLOCK (≥ 0.70 / Rule, цена FP 150).",
            "• Признаки подняли LogReg с 0.8135 до 0.9589 и снизили FP-затраты GBDT в 5 раз.",
        ],
        accent_hex="0284C7", title_pt=13.5, body_pt=11.5,
    )
    s4.add_card(
        6.82, 4.88, 5.95, 1.80,
        "Финансовый итог полного ИИ-агента (Test Set)",
        [
            "• PR-AUC 0.9745 | Precision@100 0.9900 (против 0.5622 у Rules Only).",
            "• Предотвращено 2 806 239 из 2 824 069 (99.31%), затраты FP: 1 543, чистая экономия: 2 804 696 при 13 REVIEW.",
        ],
        accent_hex="16A34A", title_pt=13.5, body_pt=11.5,
    )
    s4.add_footer()
    slides.append(s4)

    # ------------------------------------------------------------------
    # SLIDE 5: EXPLAINABILITY, DASHBOARD & MLOPS
    # ------------------------------------------------------------------
    s5 = SlideBuilder(5, total)
    s5.add_header("04 · Объяснимость и MLOps-мониторинг", "SHAP для аналитика, Streamlit-дашборд, PSI и дообучение")
    s5.add_image("../media/image3.png", ASSETS_DIR / "shap_case_example.png", 0.55, 1.48, 5.95, 3.35)
    s5.add_image("../media/image4.png", ASSETS_DIR / "psi_drift.png", 6.82, 1.48, 5.95, 3.35)
    s5.add_card(
        0.55, 5.00, 5.95, 1.68,
        "SHAP + LLM и Streamlit-дашборд",
        [
            "• Топ-5 факторов SHAP на русском языке (+ опциональное LLM-резюме строго поверх SHAP).",
            "• Дашборд: очередь алертов, разметка, PSI, абляции и ручной симулятор.",
        ],
        accent_hex="6D28D9", title_pt=13.5, body_pt=11.5,
    )
    s5.add_card(
        6.82, 5.00, 5.95, 1.68,
        "Дрейф PSI (CRITICAL) и Feedback Loop",
        [
            "• PSI поймал сдвиг в Test: dest_in_degree = 2.003, risk_score = 0.741 (схема novel_api_cashout).",
            "• Разметка (/feedback) → БД → /retrain дообучает модели с весом свежих меток 3×.",
        ],
        accent_hex="DC2626", title_pt=13.5, body_pt=11.5,
    )
    s5.add_footer()
    slides.append(s5)

    # ------------------------------------------------------------------
    # SLIDE 6: RISKS & CHECKLIST
    # ------------------------------------------------------------------
    s6 = SlideBuilder(6, total)
    s6.add_header("05 · Риски и итоги проекта", "Ограничения внедрения и готовность системы")
    s6.add_card(
        0.55, 1.50, 5.95, 5.10,
        "Ключевые риски и меры снижения",
        [
            "• Дрейф концепции → PSI-мониторинг + Isolation Forest + дообучение.",
            "• Запаздывающие метки → учёт лага подтверждения chargeback.",
            "• Синтетические данные → перед продом необходим пилот на реальных логах.",
            "• Ложные срабатывания → мягкие действия OTP и REVIEW вместо жёсткого блока.",
            "• Приватность и справедливость → псевдонимизация ID, без PII и защищённых атрибутов.",
        ],
        accent_hex="D97706", title_pt=15.0, body_pt=12.6,
    )
    s6.add_card(
        6.82, 1.50, 5.95, 5.10,
        "Итоги (выполнено 100%)",
        [
            "• Воспроизводимый репозиторий (README.md, config.yaml) и 2 ноутбука (EDA и абляции).",
            "• FastAPI-сервис (:8000), поток replay.py, Dockerfile и docker-compose.yml.",
            "• Рабочий Streamlit-дашборд (:8501) с предзаполненной БД.",
            "• Аналитический отчёт project_report.md и руководство user_guide.pdf.",
            "• Все автотесты пройдены (7/7 OK), включая проверку утечек данных.",
        ],
        accent_hex="16A34A", title_pt=15.0, body_pt=12.6,
    )
    s6.add_footer()
    slides.append(s6)

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
