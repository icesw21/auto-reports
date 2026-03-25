"""Obsidian-flavored markdown report generator using Jinja2.

Smart merge: when an existing Obsidian note exists, data sections (1-3) are
replaced with latest data, while analysis sections (4-7) are merged via LLM
to preserve user manual edits and append new information.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from auto_reports.models.report import ReportData

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

# Sections whose content is always replaced with fresh data
_DATA_SECTIONS = {"## 1. 기본사항", "## 2. 재무상태표", "## 3. 손익계산서"}

# Sections where user edits are preserved via LLM merge
_ANALYSIS_SECTIONS = {
    "## 4. 사업 모델",
    "## 5. 투자 아이디어",
    "## 6. 리스크 (Bear Case)",
    "## 7. 결론",
}


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


# ------------------------------------------------------------------
# Section-level parsing and merging
# ------------------------------------------------------------------

def _parse_sections(markdown: str) -> OrderedDict[str, str]:
    """Split markdown into ordered sections keyed by ``## `` headers.

    Returns OrderedDict where:
    - ``"__frontmatter__"`` → the ``---...---`` block (if present)
    - ``"__preamble__"`` → text between frontmatter and first ``## ``
    - ``"## N. Title"`` → section content (without the header line itself)
    """
    sections: OrderedDict[str, str] = OrderedDict()

    # Extract frontmatter
    fm_match = re.match(r"(---\s*\r?\n.*?\r?\n---\s*\r?\n?)", markdown, re.DOTALL)
    if fm_match:
        sections["__frontmatter__"] = fm_match.group(1)
        rest = markdown[fm_match.end():]
    else:
        sections["__frontmatter__"] = ""
        rest = markdown

    # Split on ## headers
    parts = re.split(r"^(## .+)$", rest, flags=re.MULTILINE)

    # parts[0] is preamble (before first ##)
    if parts:
        sections["__preamble__"] = parts[0]

    # Remaining parts alternate: header, content, header, content, ...
    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip()
        content = parts[i + 1]
        sections[header] = content
        i += 2

    return sections


def _merge_frontmatter(existing_fm: str, new_fm: str) -> str:
    """Merge frontmatter: preserve ``created``, update rest from new."""
    # Extract created from existing
    created = None
    m = re.search(r"^created:\s*(.+)$", existing_fm, re.MULTILINE)
    if m:
        created = m.group(1).strip()

    if not created:
        return new_fm

    # Replace created in new frontmatter, set updated to today
    result = new_fm
    result = re.sub(
        r"^(created:\s*)(.+)$",
        rf"\g<1>{created}",
        result,
        flags=re.MULTILINE,
    )
    today = date.today().isoformat()
    result = re.sub(
        r"^(updated:\s*)(.+)$",
        rf"\g<1>{today}",
        result,
        flags=re.MULTILINE,
    )
    return result


def _llm_merge_section(
    section_header: str,
    existing_content: str,
    new_content: str,
    api_key: str,
    model: str = "",
    base_url: str = "",
) -> str:
    """Merge an analysis section using LLM.

    Compares existing (possibly user-edited) content with newly generated
    content and produces a merged result that preserves user edits while
    incorporating new information.
    """
    from auto_reports.fetchers.rate_limiter import get_llm_limiter
    from openai import OpenAI

    if not model:
        model = "gpt-4.1-mini"

    existing_stripped = existing_content.strip()
    new_stripped = new_content.strip()

    # If existing is empty, just use new
    if not existing_stripped:
        return new_content

    # If new is empty, keep existing
    if not new_stripped:
        return existing_content

    # If identical, no merge needed
    if existing_stripped == new_stripped:
        return existing_content

    client = OpenAI(
        api_key=api_key, max_retries=3,
        **({"base_url": base_url} if base_url else {}),
    )

    prompt = (
        f"당신은 한국 주식 투자분석 리포트 편집자입니다.\n\n"
        f"아래는 기존 옵시디언 노트의 '{section_header}' 섹션과 "
        f"새로 생성된 최신 리포트의 동일 섹션입니다.\n\n"
        f"### 기존 노트 (유저가 수기 편집했을 수 있음)\n"
        f"{existing_stripped[:3000]}\n\n"
        f"### 최신 리포트\n"
        f"{new_stripped[:3000]}\n\n"
        f"### 머지 규칙\n"
        f"- 최신 리포트에서 새로 추가된 정보(신사업, 파이프라인, 새 리스크 등)는 적절한 위치에 덧붙임\n"
        f"- 기존 내용 중 수치나 상황이 변한 부분은 최신 데이터로 업데이트\n"
        f"- 기존 마크다운 서식(불릿, 테이블, 볼드 등)을 유지\n"
        f"- 섹션 헤더(## )는 포함하지 말 것 (내용만 반환)\n"
        f"- 응답은 머지된 섹션 내용만 (설명/코멘트 불필요)"
    )

    get_llm_limiter().wait()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=3000,
        )
        result = (response.choices[0].message.content or "").strip()
        if result:
            logger.info("LLM merged section: %s", section_header)
            return "\n" + result + "\n"
    except Exception as e:
        logger.warning("LLM merge failed for %s: %s — keeping existing", section_header, e)

    return existing_content


def _merge_report(
    existing_text: str,
    new_text: str,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
) -> str:
    """Merge existing Obsidian note with newly generated report.

    - Data sections (1,2,3): replaced with new content
    - Analysis sections (4,5,6,7): merged via LLM (preserves user edits)
    - Frontmatter: created preserved, updated/tags/links refreshed
    - Unknown sections (user-added): preserved at their original position
    """
    existing_sections = _parse_sections(existing_text)
    new_sections = _parse_sections(new_text)

    merged: OrderedDict[str, str] = OrderedDict()

    # Merge frontmatter
    merged["__frontmatter__"] = _merge_frontmatter(
        existing_sections.get("__frontmatter__", ""),
        new_sections.get("__frontmatter__", ""),
    )

    # Use new preamble (text between frontmatter and first ##)
    merged["__preamble__"] = new_sections.get("__preamble__", "")

    # Process sections: iterate over new sections to maintain order,
    # then append any existing-only sections (user-added)
    processed_headers: set[str] = set()

    for header, new_content in new_sections.items():
        if header in ("__frontmatter__", "__preamble__"):
            continue
        processed_headers.add(header)

        if header in _DATA_SECTIONS:
            # Data sections: always replace
            merged[header] = new_content
        elif header in _ANALYSIS_SECTIONS:
            existing_content = existing_sections.get(header, "")
            if existing_content.strip() and api_key:
                # LLM merge: preserve user edits + add new info
                merged[header] = _llm_merge_section(
                    header, existing_content, new_content,
                    api_key=api_key, model=model, base_url=base_url,
                )
            else:
                merged[header] = new_content
        else:
            # Unknown new section: use new content
            merged[header] = new_content

    # Preserve user-added sections not in new report
    for header, content in existing_sections.items():
        if header in ("__frontmatter__", "__preamble__"):
            continue
        if header not in processed_headers:
            merged[header] = content

    # Reassemble markdown
    parts: list[str] = []
    for key, content in merged.items():
        if key == "__frontmatter__":
            parts.append(content)
        elif key == "__preamble__":
            parts.append(content)
        else:
            parts.append(f"{key}\n{content}")

    return "".join(parts)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def write_report(
    data: ReportData,
    output_path: str | Path,
    force_overwrite: bool = False,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
) -> Path:
    """Generate and write report to file with smart merge.

    If output_path already exists and force_overwrite is False:
    - Data sections (1,2,3) are replaced with latest data
    - Analysis sections (4,5,6,7) are merged via LLM (user edits preserved)
    - Frontmatter created date is preserved

    Args:
        data: Report data to render.
        output_path: Destination file path.
        force_overwrite: If True, skip merge and overwrite entirely.
        api_key: LLM API key for analysis section merging.
        model: LLM model name.
        base_url: LLM base URL.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve created date in frontmatter
    if output_path.exists() and not force_overwrite:
        existing_created = _extract_frontmatter_created(output_path)
        if existing_created:
            data.frontmatter.created = existing_created
            today = date.today().isoformat()
            if not data.frontmatter.updated or data.frontmatter.updated[:10] < today:
                data.frontmatter.updated = today

    new_content = generate_report(data)

    if output_path.exists() and not force_overwrite:
        existing_text = output_path.read_text(encoding="utf-8")
        merged = _merge_report(
            existing_text, new_content,
            api_key=api_key, model=model, base_url=base_url,
        )
        output_path.write_text(merged, encoding="utf-8")
        logger.info("Report merged and written to %s", output_path)
    else:
        output_path.write_text(new_content, encoding="utf-8")
        logger.info("Report written to %s", output_path)

    return output_path
