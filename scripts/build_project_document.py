"""Build the sanitized Word report from its editable Markdown source.

Supports headings, paragraphs, pipe tables, simple inline formatting, local
figures and explicit page breaks. Never reads private context or row-level data.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/TAK861_Advanced_Therapy_Switch_Project_Report.md"


def inline(paragraph, text):
    for part in re.split(r"(\*\*.*?\*\*|\x60[^\x60]+\x60)", text):
        if not part:
            continue
        run = paragraph.add_run(
            part[2:-2]
            if part.startswith("**")
            else part[1:-1]
            if part.startswith(chr(96))
            else part
        )
        if part.startswith("**"):
            run.bold = True
        elif part.startswith(chr(96)):
            run.font.name = "Consolas"
            run.font.size = Pt(10)


def configure(document):
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(0.75)
    section.left_margin = section.right_margin = Inches(0.8)
    section.header_distance = section.footer_distance = Inches(0.3)
    for name in ("Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3"):
        style = document.styles[name]
        style.font.name = "Arial"
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.underline = False
        if style.element.pPr is not None:
            for child in list(style.element.pPr):
                if child.tag == qn("w:pBdr"):
                    style.element.pPr.remove(child)
    normal = document.styles["Normal"]
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.12
    normal.paragraph_format.widow_control = True
    document.styles["Title"].font.size = Pt(25)
    document.styles["Title"].font.bold = True
    document.styles["Title"].paragraph_format.space_after = Pt(16)
    for name, size in (("Heading 1", 16), ("Heading 2", 12), ("Heading 3", 11)):
        style = document.styles[name]
        style.font.size = Pt(size)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(13)
        style.paragraph_format.space_after = Pt(8)
        style.paragraph_format.keep_with_next = True
    header = section.header.paragraphs[0]
    header.text = "Advanced Therapy Switch Benchmark"
    header.runs[0].font.size = Pt(9)
    header.runs[0].font.color.rgb = RGBColor(0, 0, 0)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Engineering report  |  ").font.size = Pt(9)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    props = document.core_properties
    props.title = "TAK861 Advanced Therapy Switch Benchmark"
    props.subject = "Synthetic benchmark engineering and validation"
    props.author = props.last_modified_by = props.comments = ""
    props.keywords = "synthetic benchmark, advanced therapy, offline evaluation"
    props.created = props.modified = datetime(2026, 9, 10, tzinfo=timezone.utc)


def add_table(document, lines):
    records = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    records = [row for row in records if not all(re.fullmatch(r":?-+:?", c) for c in row)]
    columns = len(records[0])
    if any(len(row) != columns for row in records):
        raise ValueError("Inconsistent report table")
    if columns == 2:
        widths = [1.9, 5.0]
    elif columns == 3:
        widths = [1.9, 2.1, 2.9]
    else:
        first = 2.5 if columns <= 5 else 2.1
        widths = [first] + [(6.9 - first) / (columns - 1)] * (columns - 1)
    table = document.add_table(rows=0, cols=columns)
    table.autofit = False
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{side}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), "D9D9D9")
        borders.append(border)
    table._tbl.tblPr.append(borders)
    for i, record in enumerate(records):
        row = table.add_row()
        row_properties = row._tr.get_or_add_trPr()
        row_properties.append(OxmlElement("w:cantSplit"))
        if i == 0:
            row_properties.append(OxmlElement("w:tblHeader"))
        for j, (cell, value) in enumerate(zip(row.cells, record)):
            cell.width = Inches(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tc = cell._tc.get_or_add_tcPr()
            margin = OxmlElement("w:tcMar")
            for side in ("top", "left", "bottom", "right"):
                element = OxmlElement(f"w:{side}")
                element.set(qn("w:w"), "80" if side in {"top", "bottom"} else "95")
                element.set(qn("w:type"), "dxa")
                margin.append(element)
            tc.append(margin)
            if i == 0:
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), "E8EDF2")
                tc.append(shading)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.08
            if columns >= 4 and j > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            inline(p, value)
            for run in p.runs:
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(0, 0, 0)
                if i == 0:
                    run.bold = True
    document.add_paragraph().paragraph_format.space_after = Pt(1)


def build(source, output):
    document = Document()
    configure(document)
    lines = source.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- pagebreak -->":
            document.add_page_break()
        elif line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            add_table(document, table_lines)
            continue
        elif line.startswith("# "):
            inline(document.add_paragraph(style="Title"), line[2:])
        elif line.startswith("## "):
            inline(document.add_paragraph(style="Heading 1"), line[3:])
        elif line.startswith("### "):
            inline(document.add_paragraph(style="Heading 2"), line[4:])
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
            if not match:
                raise ValueError("Invalid figure reference")
            image = (source.parent / match[2]).resolve()
            if source.parent.resolve() not in image.parents:
                raise ValueError("Report figure must be inside docs")
            document.add_picture(str(image), width=Inches(6.7))
            caption = document.add_paragraph(match[1])
            caption.runs[0].font.size = Pt(9)
        elif line.startswith("- "):
            inline(document.add_paragraph(style="List Bullet"), line[2:])
        else:
            paragraph = [line]
            while (
                i + 1 < len(lines)
                and lines[i + 1].strip()
                and not lines[i + 1].startswith(("#", "|", "![", "<!--", "- "))
            ):
                i += 1
                paragraph.append(lines[i].strip())
            inline(document.add_paragraph(), " ".join(paragraph))
        i += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=REPORT)
    parser.add_argument("--output", type=Path, default=REPORT.with_suffix(".docx"))
    args = parser.parse_args()
    build(args.source, args.output)
