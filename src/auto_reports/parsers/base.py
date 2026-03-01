"""Base parser utilities for DART HTML disclosure pages."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from auto_reports.models.disclosure import parse_korean_number  # re-export

__all__ = [
    "extract_title",
    "table_to_dict",
    "table_to_rows",
    "parse_korean_number",
    "clean_text",
]



def extract_title(soup: BeautifulSoup) -> str:
    """Extract the disclosure title from a DART page.

    Tries, in order:
    1. <title> tag
    2. First <h1> / <h2> / <h3>
    3. First <td> or <th> with class containing 'title'
    4. First non-empty text from the page
    """
    # 1. <title> tag
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        return clean_text(title_tag.get_text())

    # 2. Heading tags
    for tag_name in ("h1", "h2", "h3"):
        heading = soup.find(tag_name)
        if heading and heading.get_text(strip=True):
            return clean_text(heading.get_text())

    # 3. Element with 'title' in class
    title_elem = soup.find(class_=re.compile(r"title", re.I))
    if title_elem and title_elem.get_text(strip=True):
        return clean_text(title_elem.get_text())

    # 4. Fallback
    body = soup.find("body")
    if body:
        text = body.get_text(separator=" ", strip=True)
        first_line = text.split("\n")[0]
        return clean_text(first_line)[:200]

    return ""


def table_to_dict(table: Tag) -> dict[str, str]:
    """Convert a DART key-value table to a dict.

    DART key-value tables have rows (<tr>) with 2+ <td> elements:
    the first is the label, the second is the value.  Rows with only 1 cell
    are treated as section headers and skipped.

    For rows with 3 or more cells (e.g. [section_header, sub_key, value]),
    the last two cells are also extracted as a key-value pair.  Existing
    keys are not overridden by this secondary extraction.

    Args:
        table: BeautifulSoup Tag for a <table> element.

    Returns:
        Dict mapping label text to value text.
    """
    result: dict[str, str] = {}
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) >= 2:
            key = clean_text(cells[0].get_text())
            value = clean_text(cells[1].get_text())
            if key:
                result[key] = value
            # For 3+ column rows (DART nested key-value tables):
            # cells[0] may be a section header or empty (rowspan sub-row),
            # with the actual key-value pair in the last two cells.
            # e.g. [전환에 따라 발행할 주식, 주식수, 877,346]
            # e.g. ["", 주식수, 877,346]
            if len(cells) >= 3:
                sub_key = clean_text(cells[-2].get_text())
                sub_value = clean_text(cells[-1].get_text())
                if sub_key and sub_key not in result:
                    result[sub_key] = sub_value
    return result


def table_to_rows(table: Tag) -> list[dict[str, str]]:
    """Convert a multi-column DART table with headers to a list of dicts.

    The first <tr> containing <th> cells (or the first <tr> overall) is used
    as the header row.  Subsequent <tr> rows become dicts keyed by header
    text.

    Args:
        table: BeautifulSoup Tag for a <table> element.

    Returns:
        List of dicts, one per data row.
    """
    rows = table.find_all("tr")
    if not rows:
        return []

    # Find header row: first row that has <th> cells, else the very first row
    header_row_idx = 0
    headers: list[str] = []
    for idx, row in enumerate(rows):
        th_cells = row.find_all("th")
        if th_cells:
            headers = [clean_text(th.get_text()) for th in th_cells]
            header_row_idx = idx
            break

    if not headers:
        # Fall back to first row's td cells as headers
        first_cells = rows[0].find_all(["td", "th"])
        headers = [clean_text(c.get_text()) for c in first_cells]
        header_row_idx = 0

    result: list[dict[str, str]] = []
    for row in rows[header_row_idx + 1 :]:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        # Skip rows that look like sub-headers (all cells have colspan or are bold)
        row_dict: dict[str, str] = {}
        for i, cell in enumerate(cells):
            key = headers[i] if i < len(headers) else f"col_{i}"
            row_dict[key] = clean_text(cell.get_text())
        result.append(row_dict)

    return result


def clean_text(text: str) -> str:
    """Strip whitespace, normalize internal spaces, remove non-printable chars.

    Args:
        text: Raw text extracted from HTML.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""
    # Collapse whitespace (including &nbsp; which becomes \xa0)
    text = re.sub(r"[\xa0\u3000\t\r\n]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
