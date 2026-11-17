#!/usr/bin/env python3
"""把本项目的 Markdown 结果报告排版为 PDF；不读取或计算模型预测。"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
from pathlib import Path
import re
import textwrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image, KeepTogether, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


def inline(text):
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#126078">\1</link>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r'<b>\1</b>', text)
    text = re.sub(r"`([^`]+)`", r'<font color="#344054">\1</font>', text)
    return text


def render(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError("report PDF already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdfmetrics.registerFontFamily("STSong-Light", normal="STSong-Light", bold="STSong-Light",
                                  italic="STSong-Light", boldItalic="STSong-Light")
    styles = getSampleStyleSheet()
    body = ParagraphStyle("ChineseBody", fontName="STSong-Light", fontSize=10.3, leading=16.3,
                          spaceAfter=7, wordWrap="CJK", alignment=TA_LEFT)
    title = ParagraphStyle("ChineseTitle", parent=body, fontSize=20, leading=27, spaceAfter=17,
                           textColor=colors.HexColor("#163544"), keepWithNext=True)
    section = ParagraphStyle("ChineseSection", parent=body, fontSize=13.5, leading=20, spaceBefore=13,
                             spaceAfter=7, textColor=colors.HexColor("#163544"), keepWithNext=True)
    subsection = ParagraphStyle("ChineseSubsection", parent=body, fontSize=11.2, leading=17,
                                spaceBefore=7, keepWithNext=True)
    cell = ParagraphStyle("ChineseCell", parent=body, fontSize=8.4, leading=12, spaceAfter=0)
    caption = ParagraphStyle("ChineseCaption", parent=body, fontSize=8.5, leading=13,
                             textColor=colors.HexColor("#3F4E56"))
    code = ParagraphStyle("Commands", fontName="Courier", fontSize=7.5, leading=10,
                          backColor=colors.HexColor("#F3F6F8"), borderPadding=7,
                          spaceBefore=5, spaceAfter=9)
    width = A4[0] - 88
    flow, lines, i = [], source.read_text(encoding="utf-8").splitlines(), 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("```"):
            block, i = [], i + 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.extend(textwrap.wrap(lines[i], width=100, replace_whitespace=False,
                                           drop_whitespace=False) or [""])
                i += 1
            flow.append(Preformatted("\n".join(block), code))
            i += 1
            continue
        if line.startswith("|"):
            table = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                fields = [part.strip() for part in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", value) for value in fields):
                    table.append(fields)
                i += 1
            columns = len(table[0])
            if any(len(row) != columns for row in table):
                raise ValueError("inconsistent Markdown table columns")
            weights = [min(30, max(8, max(len(row[j]) for row in table))) for j in range(columns)]
            column_widths = [width * value / sum(weights) for value in weights]
            obj = Table([[Paragraph(inline(value), cell) for value in row] for row in table],
                        colWidths=column_widths, repeatRows=1, hAlign="LEFT")
            obj.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF0F3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 0), (-1, 0), .65, colors.HexColor("#687A84")),
                ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor("#D5DCE0")),
            ]))
            flow.extend([obj, Spacer(1, 10)])
            continue
        picture = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if picture:
            image = Image(str((source.parent / picture[2]).resolve()))
            factor = min(width / image.imageWidth, 390 / image.imageHeight)
            image.drawWidth, image.drawHeight = image.imageWidth * factor, image.imageHeight * factor
            flow.append(KeepTogether([image, Spacer(1, 4), Paragraph(inline(picture[1]), caption), Spacer(1, 8)]))
            i += 1
            continue
        if line.startswith("### "):
            flow.append(Paragraph(inline(line[4:]), subsection))
        elif line.startswith("## "):
            flow.append(Paragraph(inline(line[3:]), section))
        elif line.startswith("# "):
            flow.append(Paragraph(inline(line[2:]), title))
        elif line.startswith("- "):
            flow.append(Paragraph("• " + inline(line[2:]), body))
        else:
            paragraph = [line]
            while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith(("#", "|", "```", "![", "- ")):
                i += 1
                paragraph.append(lines[i].strip())
            flow.append(Paragraph(inline(" ".join(paragraph)), body))
        i += 1
    short_sha = hashlib.sha256(source.read_bytes()).hexdigest()[:12]

    def page(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#637480"))
        canvas.drawString(44, 25, f"PU all-data retraining | Report source {short_sha}")
        canvas.drawRightString(A4[0] - 44, 25, str(document.page))
        canvas.setStrokeColor(colors.HexColor("#C9D2D8"))
        canvas.setLineWidth(.4)
        canvas.line(44, 39, A4[0] - 44, 39)

    document = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=44, leftMargin=44,
                                 topMargin=42, bottomMargin=53, title="PU 全数据重训练结果报告",
                                 author="CoarsePGNN research project")
    document.build(flow, onFirstPage=page, onLaterPages=page)
    print(str(destination))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(args.source, args.output)
