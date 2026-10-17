from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle)
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "taller4-ballesteros" / "informe_taller4.md"
OUTPUT = ROOT / "taller4-ballesteros" / "output" / "pdf" / "Informe_Taller4_Ballesteros.pdf"


def ascii_dashes(text: str) -> str:
    return (text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
            .replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-"))


def inline(text: str) -> str:
    text = escape(ascii_dashes(text))
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    return text


def footer(canvas, doc):
    canvas.saveState()
    width, _ = letter
    canvas.setStrokeColor(colors.HexColor("#d5deea"))
    canvas.line(0.68 * inch, 0.57 * inch, width - 0.68 * inch, 0.57 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#526174"))
    canvas.drawString(0.7 * inch, 0.38 * inch, "Taller 4 - LLMOps | Jessica Ballesteros")
    canvas.drawRightString(width - 0.7 * inch, 0.38 * inch, f"Pagina {doc.page}")
    canvas.restoreState()


def build():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold",
                              fontSize=23, leading=28, textColor=colors.HexColor("#15345b"),
                              alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle(name="H1Custom", parent=styles["Heading1"], fontSize=17, leading=21,
                              textColor=colors.HexColor("#15345b"), spaceBefore=16, spaceAfter=9,
                              keepWithNext=True))
    styles.add(ParagraphStyle(name="H2Custom", parent=styles["Heading2"], fontSize=12.5, leading=16,
                              textColor=colors.HexColor("#176b87"), spaceBefore=12, spaceAfter=6,
                              keepWithNext=True))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontSize=9.2, leading=13.2,
                              textColor=colors.HexColor("#243247"), spaceAfter=6))
    styles.add(ParagraphStyle(name="SmallCustom", parent=styles["BodyText"], fontSize=7.5, leading=9.5,
                              textColor=colors.HexColor("#243247")))
    styles.add(ParagraphStyle(name="CodeCustom", parent=styles["Code"], fontName="Courier", fontSize=7.2,
                              leading=9, backColor=colors.HexColor("#f1f5f9"), borderPadding=6,
                              spaceBefore=4, spaceAfter=7))

    doc = BaseDocTemplate(str(OUTPUT), pagesize=letter, leftMargin=.7*inch, rightMargin=.7*inch,
                          topMargin=.65*inch, bottomMargin=.75*inch,
                          title="Informe Taller 4 - LLMOps", author="Jessica Ballesteros")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=footer)])
    lines = ascii_dashes(SOURCE.read_text(encoding="utf-8")).splitlines()
    story = []
    i = 0
    first_heading = True
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i].rstrip())
                i += 1
            story.append(Preformatted("\n".join(code), styles["CodeCustom"], maxLineLength=105))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\- ]+", c or "-") for c in cells):
                    rows.append(cells)
                i += 1
            data = [[Paragraph(inline(c), styles["SmallCustom"]) for c in row] for row in rows]
            if data:
                col_count = max(map(len, data))
                widths = [doc.width / col_count] * col_count
                table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5edf6")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#15345b")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#c9d4e2")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
                ]))
                story.extend([Spacer(1, 4), table, Spacer(1, 7)])
            continue
        if line.startswith("# "):
            title = inline(line[2:])
            story.append(Paragraph(title, styles["TitleCustom"] if first_heading else styles["H1Custom"]))
            first_heading = False
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["H1Custom"]))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["H2Custom"]))
            i += 1
            continue
        if line.startswith("---"):
            story.append(Spacer(1, 5))
            i += 1
            continue
        if line.startswith(("* ", "- ")) or re.match(r"\d+\. ", line):
            match = re.match(r"(?:\*|-|\d+\.)\s+(.*)", line)
            story.append(Paragraph("&#8226; " + inline(match.group(1)), styles["BodyCustom"]))
            i += 1
            continue
        if line.startswith("> "):
            story.append(Paragraph(inline(line[2:]), styles["BodyCustom"]))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(("#", "|", "* ", "- ", "```", "> ")) and not re.match(r"\s*\d+\. ", lines[i]):
            paragraph.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline(" ".join(paragraph)), styles["BodyCustom"]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
