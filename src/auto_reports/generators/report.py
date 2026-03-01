"""Obsidian-flavored markdown report generator using Jinja2."""

from __future__ import annotations

import logging
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


def write_report(data: ReportData, output_path: str | Path) -> Path:
    """Generate and write report to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = generate_report(data)
    output_path.write_text(content, encoding="utf-8")
    logger.info("Report written to %s", output_path)
    return output_path
