"""
Collectors package for DART research report collector.

This package contains collector classes for different data sources:
- DartCollector: DART regulatory filings via API
- FnGuideCollector: FnGuide research reports via Selenium
- NaverCollector: Naver Finance research reports via HTTP
- NewsCollector: Financial news from Naver Finance and Google News RSS
"""

from auto_reports.collectors.base import BaseCollector
from auto_reports.collectors.dart_collector import DartCollector
from auto_reports.collectors.fnguide_collector import FnGuideCollector
from auto_reports.collectors.naver_collector import NaverCollector
from auto_reports.collectors.news_collector import NewsCollector

__all__ = [
    'BaseCollector',
    'DartCollector',
    'FnGuideCollector',
    'NaverCollector',
    'NewsCollector',
]
