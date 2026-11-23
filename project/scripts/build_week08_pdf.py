#!/usr/bin/env python3
"""Render a plain-text JSON course report as Chinese A4 PDF and editable Markdown.

The caller must perform the PDF skill's artifact-operation marker immediately
before the actual PDF-authoring invocation. --check-fonts creates no artifacts.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image as PillowImage, ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FONT_NAME = "Week08Chinese"
FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/msyh.ttc"),
)
PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = RIGHT_MARGIN = 42.0
TOP_MARGIN = 55.0
BOTTOM_MARGIN = 43.0
CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
# ReportLab frames reserve 6 pt padding on each side by default.
FRAME_WIDTH = CONTENT_WIDTH - 12.0
FRAME_HEIGHT = PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN - 12.0
INK = colors.HexColor("#17212D")
MUTED = colors.HexColor("#586573")
ACCENT = colors.HexColor("#174E68")
RULE = colors.HexColor("#D5DFE5")
PALE = colors.HexColor("#EEF3F6")


def register_font(explicit_path: Path | None = None) -> Path:
    """Register a subsettable local font, rejecting missing Chinese glyphs."""
    failures = []
    configured = explicit_path or (Path(os.environ["COURSE_REPORT_FONT"]) if os.environ.get("COURSE_REPORT_FONT") else None)
    candidates = (configured,) if configured else FONT_CANDIDATES
    for path in candidates:
        if not path.is_file():
            failures.append(f"{path}: not found")
            continue
        try:
            font = TTFont(FONT_NAME, str(path), subfontIndex=0)
            required = "计算机视觉课程实验中文图像"
            missing = [c for c in required if ord(c) not in font.face.charToGlyph]
            if missing:
                raise ValueError("missing Chinese glyphs: " + "".join(missing))
            pdfmetrics.registerFont(font)
            # A single embedded face is deliberately used for every style.
            pdfmetrics.registerFontFamily(
                FONT_NAME,
                normal=FONT_NAME,
                bold=FONT_NAME,
                italic=FONT_NAME,
                boldItalic=FONT_NAME,
            )
            if pdfmetrics.stringWidth(required, FONT_NAME, 10) <= 0:
                raise ValueError("invalid glyph metrics")
            return path
        except Exception as exc:
            failures.append(f"{path}: {exc}")
    raise RuntimeError("No usable embeddable Chinese font.\n" + "\n".join(failures))


def plain(value: Any) -> str:
    text = "" if value is None else str(value)
    # Keep report typography compatible with the PDF skill's ASCII-hyphen rule.
    return text.translate(str.maketrans({c: "-" for c in "‐‑‒–—−"}))


def paragraph_text(value: Any) -> str:
    """Input is always text, never trusted ReportLab XML or HTML."""
    return html.escape(plain(value), quote=False).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>")


def markdown_text(value: Any, table: bool = False) -> str:
    text = html.escape(plain(value), quote=False)
    text = text.replace("\\", "\\\\")
    text = re.sub(r"([`*_\[\]{}#!])", r"\\\1", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if table:
        return text.replace("|", "\\|").replace("\n", "<br>")
    return text.replace("\n", "  \n")


def styles(paper: bool = False) -> dict[str, ParagraphStyle]:
    base = dict(fontName=FONT_NAME, textColor=INK, wordWrap="CJK", allowWidows=0, allowOrphans=0)
    result = {
        "title": ParagraphStyle("ReportTitle", fontSize=22, leading=30, spaceAfter=12, keepWithNext=True, **base),
        "subtitle": ParagraphStyle("ReportSubtitle", fontSize=11, leading=17, textColor=MUTED, fontName=FONT_NAME, wordWrap="CJK", spaceAfter=5, keepWithNext=True),
        "date": ParagraphStyle("ReportDate", fontSize=9, leading=14, textColor=MUTED, fontName=FONT_NAME, wordWrap="CJK", spaceAfter=14, keepWithNext=True),
        "h1": ParagraphStyle("Section", fontSize=15, leading=22, spaceBefore=14, spaceAfter=8, keepWithNext=True, **base),
        "h2": ParagraphStyle("Subsection", fontSize=11.5, leading=17, spaceBefore=10, spaceAfter=6, keepWithNext=True, **base),
        "p": ParagraphStyle("Body", fontSize=10, leading=16, spaceAfter=8, alignment=TA_LEFT, **base),
        "bullet": ParagraphStyle("Bullet", fontSize=10, leading=16, leftIndent=12, firstLineIndent=-10, spaceAfter=5, **base),
        "table": ParagraphStyle("TableBody", fontSize=8.3, leading=12, spaceAfter=0, **base),
        "table_header": ParagraphStyle("TableHeader", fontSize=8.5, leading=12, textColor=ACCENT, fontName=FONT_NAME, wordWrap="CJK", spaceAfter=0),
        "caption": ParagraphStyle("Caption", fontSize=8.5, leading=12, textColor=MUTED, fontName=FONT_NAME, wordWrap="CJK", alignment=TA_CENTER, spaceBefore=5, spaceAfter=9),
        "reference": ParagraphStyle("Reference", fontSize=8, leading=11, spaceAfter=5, **base),
    }
    if paper:
        result["title"].fontSize = 19
        result["title"].leading = 27
        result["title"].spaceAfter = 8
        result["subtitle"].fontSize = 10
        result["date"].spaceAfter = 10
        result["h1"].fontSize = 13
        result["h1"].leading = 19
        result["h1"].spaceBefore = 11
        result["h2"].fontSize = 11
        result["h2"].spaceBefore = 8
        result["caption"].alignment = TA_LEFT
        result["caption"].fontSize = 8.3
    return result


def positive_number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return default
    return number if math.isfinite(number) and number > 0 else default


def make_table(block: dict[str, Any], s: dict[str, ParagraphStyle]) -> Table:
    headers = block.get("headers", [])
    rows = block.get("rows", [])
    if not isinstance(headers, list) or not headers:
        raise ValueError("table.headers must be a non-empty list")
    if not isinstance(rows, list) or any(not isinstance(row, list) or len(row) != len(headers) for row in rows):
        raise ValueError("table rows must have the same number of columns as headers")
    raw_widths = block.get("widths")
    if raw_widths is None:
        raw_widths = [1] * len(headers)
    if not isinstance(raw_widths, list) or len(raw_widths) != len(headers):
        raise ValueError("table.widths must match the header column count")
    weights = [positive_number(item, 1.0) for item in raw_widths]
    widths = [FRAME_WIDTH * item / sum(weights) for item in weights]
    values = [[Paragraph(paragraph_text(item), s["table_header"]) for item in headers]]
    values.extend([[Paragraph(paragraph_text(item), s["table"]) for item in row] for row in rows])
    result = Table(values, colWidths=widths, repeatRows=1, splitByRow=1, hAlign="LEFT", spaceBefore=3, spaceAfter=11)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFCFD")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
    ]))
    return result


def resolve_image(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Report image not found: {path}")
    return path.resolve()


def make_image(block: dict[str, Any], s: dict[str, ParagraphStyle]) -> KeepTogether:
    path = resolve_image(block.get("path", ""))
    caption = Paragraph(paragraph_text(block.get("caption", "")), s["caption"])
    _, caption_height = caption.wrap(FRAME_WIDTH, FRAME_HEIGHT)
    image_source: Any = str(path)
    oriented_buffer = None
    with PillowImage.open(path) as source:
        orientation = source.getexif().get(274, 1)
        if orientation != 1:
            oriented = ImageOps.exif_transpose(source)
            pixel_width, pixel_height = oriented.size
            oriented_buffer = io.BytesIO()
            oriented.save(oriented_buffer, format="PNG")
            oriented_buffer.seek(0)
            image_source = oriented_buffer
        else:
            pixel_width, pixel_height = source.size
    if pixel_width <= 0 or pixel_height <= 0:
        raise ValueError(f"Invalid image dimensions: {path}")
    max_width = min(FRAME_WIDTH, positive_number(block.get("width"), FRAME_WIDTH))
    max_height = min(FRAME_HEIGHT - caption_height - 28, positive_number(block.get("max_height"), FRAME_HEIGHT - 90))
    if max_height <= 0:
        raise ValueError("Image caption is too tall for a page")
    scale = min(max_width / pixel_width, max_height / pixel_height)
    picture = Image(image_source, width=pixel_width * scale, height=pixel_height * scale, hAlign="CENTER")
    if oriented_buffer is not None:
        picture._report_image_buffer = oriented_buffer
    return KeepTogether([Spacer(1, 3), picture, caption])


def draw_page(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT_NAME, 8)
    canvas.drawString(LEFT_MARGIN, PAGE_HEIGHT - 31, "计算机视觉课程 | 前8周阶段实践")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(LEFT_MARGIN, PAGE_HEIGHT - 39, PAGE_WIDTH - RIGHT_MARGIN, PAGE_HEIGHT - 39)
    canvas.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, 25, f"第 {doc.page} 页")
    canvas.restoreState()


def make_markdown(data: dict[str, Any], markdown_path: Path | None = None) -> str:
    parts = []
    if data.get("title"):
        parts.append("# " + markdown_text(data["title"]))
    if data.get("subtitle"):
        parts.append(markdown_text(data["subtitle"]))
    if data.get("date"):
        parts.append(markdown_text(data["date"]))
    for block in data["blocks"]:
        kind = block.get("type")
        if kind in ("title", "h1", "h2"):
            level = {"title": "# ", "h1": "## ", "h2": "### "}[kind]
            parts.append(level + markdown_text(block.get("text", "")))
        elif kind in ("p", "reference"):
            parts.append(markdown_text(block.get("text", "")))
        elif kind == "bullet":
            parts.append("- " + markdown_text(block.get("text", "")).replace("  \n", "  \n  "))
        elif kind == "table":
            headers = block.get("headers", [])
            rows = block.get("rows", [])
            lines = ["| " + " | ".join(markdown_text(item, table=True) for item in headers) + " |"]
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            lines.extend("| " + " | ".join(markdown_text(item, table=True) for item in row) + " |" for row in rows)
            parts.append("\n".join(lines))
        elif kind == "image":
            path = resolve_image(block.get("path", ""))
            caption = markdown_text(block.get("caption", ""))
            link = os.path.relpath(path, markdown_path.parent) if markdown_path else str(path)
            parts.append(f"![{caption}](<{link}>)" + (f"\n\n{caption}" if caption else ""))
        elif kind == "pagebreak":
            parts.append("---")
    return "\n\n".join(parts).rstrip() + "\n"


def build_report(data: dict[str, Any], output: Path, markdown: Path) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("blocks"), list):
        raise ValueError("Report JSON must be an object containing a blocks list")
    if any(not isinstance(block, dict) for block in data["blocks"]):
        raise ValueError("Every report block must be an object")
    s = styles(data.get("document_style") == "paper")
    story = []
    for key in ("title", "subtitle", "date"):
        if data.get(key):
            story.append(Paragraph(paragraph_text(data[key]), s[key]))
    for index, block in enumerate(data["blocks"]):
        kind = block.get("type")
        if kind in ("title", "h1", "h2", "p", "bullet", "reference"):
            value = block.get("text", "")
            if kind == "bullet":
                value = "• " + plain(value)
            story.append(Paragraph(paragraph_text(value), s[kind]))
        elif kind == "table":
            story.append(make_table(block, s))
        elif kind == "image":
            story.append(make_image(block, s))
        elif kind == "pagebreak":
            story.append(PageBreak())
        else:
            # Unknown types and extra object keys are intentionally ignored.
            print(f"Ignoring unknown block type at index {index}: {kind!r}", file=sys.stderr)
    if not story:
        raise ValueError("Report has no renderable content")
    markdown_source = make_markdown(data, markdown)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    partial_pdf = output.with_name(output.stem + ".partial.pdf")
    partial_md = markdown.with_name(markdown.name + ".partial")
    try:
        doc = SimpleDocTemplate(
            str(partial_pdf),
            pagesize=A4,
            rightMargin=RIGHT_MARGIN,
            leftMargin=LEFT_MARGIN,
            topMargin=TOP_MARGIN,
            bottomMargin=BOTTOM_MARGIN,
            title=plain(data.get("title", "计算机视觉课程实验")),
            author="计算机视觉课程实验",
            pageCompression=1,
            allowSplitting=1,
        )
        doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
        partial_md.write_text(markdown_source, encoding="utf-8")
        os.replace(partial_pdf, output)
        os.replace(partial_md, markdown)
    finally:
        partial_pdf.unlink(missing_ok=True)
        partial_md.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Report JSON input")
    parser.add_argument("--output", type=Path, help="PDF output path")
    parser.add_argument("--markdown", type=Path, help="Editable Markdown output path")
    parser.add_argument("--check-fonts", action="store_true", help="Register/check Chinese fonts without generating a PDF")
    parser.add_argument("--font", type=Path, help="Optional embeddable Chinese TrueType font (also COURSE_REPORT_FONT)")
    args = parser.parse_args()
    font_path = register_font(args.font)
    if args.check_fonts:
        print(json.dumps({"font_registration": "ok", "font_name": FONT_NAME, "font_path": str(font_path), "pdf_created": False}, ensure_ascii=False))
        return 0
    if not args.data or not args.output or not args.markdown:
        parser.error("--data, --output and --markdown are required unless --check-fonts is used")
    if args.output.resolve() == args.markdown.resolve():
        parser.error("PDF and Markdown output paths must differ")
    data = json.loads(args.data.read_text(encoding="utf-8"))
    build_report(data, args.output.resolve(), args.markdown.resolve())
    print(json.dumps({"pdf": str(args.output.resolve()), "markdown": str(args.markdown.resolve()), "font_path": str(font_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
