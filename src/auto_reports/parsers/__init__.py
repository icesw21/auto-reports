"""HTML parsers for DART disclosure pages."""

from auto_reports.parsers.base import (
    clean_text,
    extract_title,
    parse_korean_number,
    table_to_dict,
    table_to_rows,
)
from auto_reports.parsers.classifier import DisclosureType, classify_disclosure
from auto_reports.parsers.contract import parse_contract
from auto_reports.parsers.convert import parse_convert
from auto_reports.parsers.convert_price import parse_convert_price
from auto_reports.parsers.exchange_disclosure import parse_exchange_disclosure
from auto_reports.parsers.issue import parse_issue, parse_rights_issue_pdf
from auto_reports.parsers.performance import parse_performance

__all__ = [
    # base utilities
    "extract_title",
    "table_to_dict",
    "table_to_rows",
    "parse_korean_number",
    "clean_text",
    # classifier
    "DisclosureType",
    "classify_disclosure",
    # parsers
    "parse_convert",
    "parse_convert_price",
    "parse_contract",
    "parse_performance",
    "parse_issue",
    "parse_rights_issue_pdf",
    "parse_exchange_disclosure",
]
