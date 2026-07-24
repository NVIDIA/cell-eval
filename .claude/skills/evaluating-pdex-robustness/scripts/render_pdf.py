#!/usr/bin/env python
"""Render a robustness_report.md to PDF (reportlab Platypus).

Handles the report's markdown subset: headings (#/##/###), the pipe tables, bullet
lists, **bold**, `code`, embedded images (![alt](relpath), e.g. the QQ-plots), and
the metric-definitions appendix (with Unicode math, via DejaVu fonts).

    uv run python scripts/render_pdf.py <report.md> <out.pdf> ["Title"]
"""

from __future__ import annotations

import glob
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

EMOJI = {"✅": "[PASS]", "⚠️": "[WARN]", "❌": "[FAIL]", "⏭️": "[SKIP]", "ℹ️": "[INFO]", "🤖": ""}

# Register DejaVu (ships with matplotlib) for full Unicode math; fall back to Helvetica.
BASE, BOLD, MONO = "Helvetica", "Helvetica-Bold", "Courier"
try:
    import matplotlib

    fdir = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(fdir, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(fdir, "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuMono", os.path.join(fdir, "DejaVuSansMono.ttf")))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")
    BASE, BOLD, MONO = "DejaVu", "DejaVu-Bold", "DejaVuMono"
except Exception:  # pragma: no cover - fonts optional
    pass


def esc(s: str) -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for k, v in EMOJI.items():
        s = s.replace(k, v)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", rf'<font face="{MONO}" size=8>\1</font>', s)
    return s


def build(md_path: str, pdf_path: str, title: str = "report") -> None:
    md_dir = os.path.dirname(os.path.abspath(md_path))
    lines = open(md_path).read().splitlines()
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName=BOLD, fontSize=18, spaceAfter=10, alignment=TA_LEFT)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName=BOLD, fontSize=13, spaceBefore=12,
                        spaceAfter=4, textColor=colors.HexColor("#1a3c6e"))
    h3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName=BOLD, fontSize=11, spaceBefore=8,
                        spaceAfter=3, textColor=colors.HexColor("#27496d"))
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName=BASE, fontSize=9.5, leading=13, spaceAfter=4)
    cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10, spaceAfter=0)
    cellh = ParagraphStyle("cellh", parent=cell, fontName=BOLD, textColor=colors.white)

    flow: list = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        img = re.match(r"!\[(.*?)\]\((.+?)\)", ln.strip())
        if ln.startswith("# "):
            flow.append(Paragraph(esc(ln[2:]), h1))
        elif ln.startswith("### "):
            flow.append(Paragraph(esc(ln[4:]), h3))
        elif ln.startswith("## "):
            flow.append(Paragraph(esc(ln[3:]), h2))
        elif img:
            path = img.group(2)
            ap = path if os.path.isabs(path) else os.path.join(md_dir, path)
            if os.path.exists(ap):
                from reportlab.lib.utils import ImageReader

                iw, ih = ImageReader(ap).getSize()
                w = min(3.6 * inch, 7.0 * inch)
                flow.append(Spacer(1, 4))
                flow.append(Image(ap, width=w, height=w * ih / iw))
                flow.append(Spacer(1, 6))
        elif ln.lstrip().startswith("|"):
            tbl = []
            while i < n and lines[i].lstrip().startswith("|"):
                tbl.append(lines[i]); i += 1
            i -= 1
            rows = []
            for raw in tbl:
                cells = [c.strip() for c in raw.strip().strip("|").split("|")]
                if set("".join(cells)) <= set("-: "):
                    continue
                rows.append([Paragraph(esc(c), cellh if not rows else cell) for c in cells])
            if rows:
                ncol = max(len(r) for r in rows)
                avail = 7.0 * inch
                widths = [avail * (0.34 if c == 0 else 0.66 / (ncol - 1)) for c in range(ncol)]
                t = Table(rows, colWidths=widths, repeatRows=1)
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c6e")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b0b0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                flow.append(t); flow.append(Spacer(1, 6))
        elif ln.strip().startswith("- "):
            items = []
            while i < n and lines[i].strip().startswith("- "):
                items.append(ListItem(Paragraph(esc(lines[i].strip()[2:]), body), leftIndent=12))
                i += 1
            i -= 1
            flow.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=14))
        elif ln.strip() == "":
            flow.append(Spacer(1, 4))
        else:
            flow.append(Paragraph(esc(ln), body))
        i += 1

    doc = SimpleDocTemplate(pdf_path, pagesize=letter, title=title,
                            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    doc.build(flow)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    md = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else md.replace(".md", ".pdf")
    title = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(out)
    build(md, out, title)
    _ = glob  # silence unused if image glob not needed
