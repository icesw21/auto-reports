"""Tests for disclosure type classifier."""

import pytest

from auto_reports.parsers.classifier import DisclosureType, classify_disclosure


@pytest.mark.parametrize(
    "title, expected",
    [
        ("전환청구권ㆍ신주인수권ㆍ교환청구권행사", DisclosureType.CONVERT),
        ("전환청구권행사", DisclosureType.CONVERT),

        ("신주인수권행사", DisclosureType.CONVERT),
        ("주식매수선택권행사", DisclosureType.CONVERT),
        ("전환가액ㆍ신주인수권행사가액ㆍ교환가액의조정", DisclosureType.CONVERT_PRICE_CHANGE),
        ("신주인수권행사가액의조정", DisclosureType.CONVERT_PRICE_CHANGE),
        ("전환가액의조정", DisclosureType.CONVERT_PRICE_CHANGE),
        ("단일판매ㆍ공급계약체결", DisclosureType.CONTRACT),
        ("단일판매ㆍ공급계약해지", DisclosureType.CONTRACT),
        ("매출액또는손익구조30%(대규모법인은15%)이상변동", DisclosureType.PERFORMANCE),
        ("전환사채권 발행결정", DisclosureType.ISSUE),

        ("신주인수권부사채권 발행결정", DisclosureType.ISSUE),
        ("유상증자결정", DisclosureType.RIGHTS_ISSUE),
        ("유무상증자결정", DisclosureType.RIGHTS_ISSUE),
    ],
)
def test_classify_known_types(title: str, expected: DisclosureType):
    assert classify_disclosure(title) == expected


def test_classify_unknown():
    assert classify_disclosure("감사보고서제출") == DisclosureType.UNKNOWN
    assert classify_disclosure("신규시설투자등") == DisclosureType.UNKNOWN


def test_classify_with_company_prefix():
    """DART titles often include company name and date."""
    title = "선익시스템/전환가액의조정/2025.12.30"
    assert classify_disclosure(title) == DisclosureType.CONVERT_PRICE_CHANGE


def test_classify_whitespace_normalization():
    assert classify_disclosure("전환 청구권 행사") == DisclosureType.CONVERT
