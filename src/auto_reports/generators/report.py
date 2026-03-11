"""Obsidian-flavored markdown report generator using Jinja2."""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from auto_reports.models.report import ReportData

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate_report(data: ReportData) -> str:
    """Generate a complete Obsidian markdown report from ReportData."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    template = env.get_template("report.md.j2")
    return template.render(data=data)


def _extract_frontmatter_created(path: Path) -> str | None:
    """기존 보고서에서 created 날짜 추출."""
    try:
        text = path.read_text(encoding="utf-8")
        m = re.match(r"---\s*\r?\n(.*?)\r?\n---", text, re.DOTALL)
        if m:
            for line in m.group(1).splitlines():
                if line.strip().startswith("created:"):
                    val = line.split(":", 1)[1].strip().strip("\"'")
                    if val:
                        return val
    except Exception:
        pass
    return None


def write_report(data: ReportData, output_path: str | Path) -> Path:
    """Generate and write report to file.

    If output_path already exists, preserves the original ``created`` date
    and sets ``updated`` to today.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 기존 파일이 있으면 created 날짜 보존 + updated 갱신
    if output_path.exists():
        existing_created = _extract_frontmatter_created(output_path)
        if existing_created:
            data.frontmatter.created = existing_created
            today = date.today().isoformat()
            if not data.frontmatter.updated or data.frontmatter.updated[:10] < today:
                data.frontmatter.updated = today

    content = generate_report(data)
    output_path.write_text(content, encoding="utf-8")
    logger.info("Report written to %s", output_path)
    return output_path
