"""Classify DART disclosures by their title into parser types."""

from __future__ import annotations

import re as _re
from enum import Enum


class DisclosureType(Enum):
    CONVERT = "convert"
    CONVERT_PRICE_CHANGE = "convert_price_change"
    CONTRACT = "contract"
    PERFORMANCE = "performance"
    ISSUE = "issue"
    RIGHTS_ISSUE = "rights_issue"
    UNKNOWN = "unknown"


# Ordered from most-specific to least-specific so that longer patterns match first.
_PATTERNS: list[tuple[str, DisclosureType]] = [
    # CONVERT_PRICE_CHANGE — must come before CONVERT because some titles share substrings
    ("전환가액ㆍ신주인수권행사가액ㆍ교환가액의조정", DisclosureType.CONVERT_PRICE_CHANGE),
    ("신주인수권행사가액의조정", DisclosureType.CONVERT_PRICE_CHANGE),
    ("전환가액의조정", DisclosureType.CONVERT_PRICE_CHANGE),
    # CONVERT
    ("전환청구권ㆍ신주인수권ㆍ교환청구권행사", DisclosureType.CONVERT),
    ("전환청구권행사", DisclosureType.CONVERT),

    ("신주인수권행사", DisclosureType.CONVERT),
    ("주식매수선택권행사", DisclosureType.CONVERT),
    # CONTRACT
    ("단일판매ㆍ공급계약체결", DisclosureType.CONTRACT),
    ("단일판매ㆍ공급계약해지", DisclosureType.CONTRACT),
    # PERFORMANCE
    ("매출액또는손익구조30%(대규모법인은15%)이상변동", DisclosureType.PERFORMANCE),
    ("매출액또는손익구조", DisclosureType.PERFORMANCE),
    # ISSUE
    ("전환사채권 발행결정", DisclosureType.ISSUE),

    ("신주인수권부사채권 발행결정", DisclosureType.ISSUE),
    ("자본으로인정되는채무증권 발행결정", DisclosureType.ISSUE),
    ("자본으로인정되는채무증권발행결정", DisclosureType.ISSUE),
    ("유무상증자결정", DisclosureType.RIGHTS_ISSUE),
    ("유상증자결정", DisclosureType.RIGHTS_ISSUE),
]


def _normalise(text: str) -> str:
    return _re.sub(r"\s+", "", text)


def classify_disclosure(title: str) -> DisclosureType:
    """Classify a DART disclosure title into a DisclosureType.

    Matching uses substring search after stripping all whitespace from both
    the title and the pattern so that minor formatting differences are ignored.
    Patterns are checked in order from most-specific to least-specific.

    Args:
        title: The disclosure title string (공시 제목).

    Returns:
        The matching DisclosureType, or DisclosureType.UNKNOWN if no pattern
        matches.
    """
    normalised = _normalise(title)
    for pattern, dtype in _PATTERNS:
        if _normalise(pattern) in normalised:
            return dtype
    return DisclosureType.UNKNOWN
