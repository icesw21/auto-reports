"""
News collector module for financial market news aggregation.

This module provides the NewsCollector class for collecting latest news
from Naver Finance and Google News RSS feeds.
"""

import os
import json
import re
import time
import warnings
from datetime import datetime, timedelta
from typing import List, Dict
from urllib.parse import quote

import requests
import urllib3
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

try:
    import feedparser
except ImportError:
    feedparser = None

try:
    from googlenewsdecoder import new_decoderv1 as _gnews_decode
except ImportError:
    _gnews_decode = None

from auto_reports.collectors.base import BaseCollector
from auto_reports.utils.retry import retry_request

# Suppress InsecureRequestWarning for news sites with expired SSL certs
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)


class _WeakDHAdapter(HTTPAdapter):
    """HTTPAdapter that tolerates servers with weak Diffie-Hellman keys."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


class NewsCollector(BaseCollector):
    """
    Collector for financial news from multiple sources.

    Collects news from:
    - Naver Finance: Company-specific news
    - Google News RSS: General market news
    - Hankyung Consensus: Analyst reports and consensus data
    - Yonhap Infomax: Financial professional news
    - Maeil Business Newspaper (MK): Securities section news

    Attributes:
        headers: HTTP headers for requests
        news_days_back: Number of days back to collect news (default: 90)

    Example:
        >>> collector = NewsCollector(news_days_back=90)
        >>> count = collector.collect(
        ...     stock_code='178320',
        ...     company_name='서진시스템'
        ... )
    """

    def __init__(self, news_days_back: int = 90, output_dir: str = None):
        """
        Initialize the news collector.

        Args:
            news_days_back: Number of days back to collect news (default: 90)
            output_dir: Output directory for collected data
        """
        super().__init__(output_dir=output_dir)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.news_days_back = news_days_back
        self.cutoff_date = datetime.now() - timedelta(days=news_days_back)

    def collect(self, stock_code: str, company_name: str, **kwargs) -> int:
        """
        Collect news for a given company from multiple sources.

        Args:
            stock_code: 6-digit stock code (e.g., '005930')
            company_name: Company name for folder organization
            **kwargs: Additional parameters (unused)

        Returns:
            int: Total number of news articles collected
        """
        self.logger.info(f"News collection started: {company_name}")
        self.logger.info(f"Collecting news for {company_name}")

        # Setup save directory
        save_dir = os.path.join(self.output_dir, company_name)
        self._ensure_directory(save_dir)

        all_news = []

        # Collect from Naver Finance
        try:
            naver_news = self._collect_naver_news(stock_code, company_name)
            all_news.extend(naver_news)
            self.logger.info(f"Naver Finance news collected: {len(naver_news)} articles")
        except Exception as e:
            self.logger.warning(f"Naver Finance news collection failed: {e}")

        # Collect from Google News RSS
        try:
            google_news = self._collect_google_news(company_name)
            all_news.extend(google_news)
            self.logger.info(f"Google News collected: {len(google_news)} articles")
        except Exception as e:
            self.logger.warning(f"Google News collection failed: {e}")

        # Collect from Hankyung Consensus
        try:
            hankyung_news = self._collect_hankyung_consensus(stock_code, company_name)
            all_news.extend(hankyung_news)
            self.logger.info(f"Hankyung consensus collected: {len(hankyung_news)} items")
        except Exception as e:
            self.logger.warning(f"Hankyung consensus collection failed: {e}")

        # Collect from TheBell
        try:
            thebell_news = self._collect_thebell_news(company_name)
            all_news.extend(thebell_news)
            self.logger.info(f"TheBell news collected: {len(thebell_news)} articles")
        except Exception as e:
            self.logger.warning(f"TheBell news collection failed: {e}")

        # Collect from Yonhap Infomax
        try:
            einfomax_news = self._collect_einfomax_news(company_name)
            all_news.extend(einfomax_news)
            self.logger.info(f"Einfomax news collected: {len(einfomax_news)} articles")
        except Exception as e:
            self.logger.warning(f"Einfomax news collection failed: {e}")

        # Collect from Maeil Business Newspaper (MK)
        try:
            mk_news = self._collect_mk_news(company_name)
            all_news.extend(mk_news)
            self.logger.info(f"MK news collected: {len(mk_news)} articles")
        except Exception as e:
            self.logger.warning(f"MK news collection failed: {e}")

        # Remove duplicates by URL
        unique_news = {news['url']: news for news in all_news}.values()
        unique_news = list(unique_news)

        # Fetch article content for top articles
        content_limit = 20
        fetched = 0
        for news_item in unique_news:
            if fetched >= content_limit:
                break
            content = self._fetch_article_content(
                news_item['url'], news_item.get('source', ''),
            )
            if content:
                news_item['content'] = content
                fetched += 1
            if fetched > 0 and fetched % 5 == 0 and fetched < content_limit:
                time.sleep(0.5)
        self.logger.info(f"Fetched article content for {fetched}/{len(unique_news)} articles")

        # Save to JSON file with today's date
        if unique_news:
            today = datetime.now().strftime('%Y%m%d')

            # 1. Save JSON file (structured data)
            news_file = os.path.join(save_dir, f"{company_name}_news_{today}.json")
            with open(news_file, 'w', encoding='utf-8') as f:
                json.dump(unique_news, f, ensure_ascii=False, indent=2)
            self.logger.info(f"News saved to {news_file}: {len(unique_news)} articles")

            # 2. Save links-only txt file (for easy copying)
            txt_dir = os.path.join(save_dir, 'txt')
            self._ensure_directory(txt_dir)
            news_links_file = os.path.join(txt_dir, f"{company_name}_news_{today}(copy).txt")
            with open(news_links_file, 'w', encoding='utf-8') as f:
                urls = [news['url'] for news in unique_news]
                f.write("\n".join(urls))
            self.logger.info(f"News links saved to {news_links_file}: {len(unique_news)} URLs")

        return len(unique_news)

    def _resolve_google_news_url(self, gnews_url: str) -> str:
        """Resolve actual article URL from a Google News RSS redirect URL."""
        if _gnews_decode is None:
            return ""
        try:
            result = _gnews_decode(gnews_url, interval=0.3)
            if result and result.get('status'):
                return result.get('decoded_url', '')
        except Exception:
            pass
        return ""

    def _fetch_article_content(self, url: str, source: str, max_chars: int = 1500) -> str:
        """Fetch and extract article body text from a news URL.

        Args:
            url: Article URL (may be relative for Naver).
            source: News source identifier (naver, google, einfomax, mk, etc.).
            max_chars: Maximum characters to return.

        Returns:
            Extracted article text, or empty string on failure.
        """
        if source in ('hankyung_consensus', 'thebell'):
            return ""

        try:
            actual_url = url

            # Resolve Google News redirect URLs
            if source == 'google' or 'news.google.com' in url:
                actual_url = self._resolve_google_news_url(url)
                if not actual_url:
                    return ""

            # Resolve Naver relative URLs
            if source == 'naver' and not url.startswith('http'):
                actual_url = f"https://finance.naver.com{url}"

            resp = retry_request(
                requests.get,
                max_retries=2,
                base_delay=0.5,
                logger=self.logger,
                url=actual_url,
                headers=self.headers,
                timeout=10,
                verify=False,
            )

            # Handle encoding
            ct = resp.headers.get('Content-Type', '')
            if 'euc-kr' in ct.lower() or 'euc_kr' in ct.lower():
                resp.encoding = 'euc-kr'
            elif resp.apparent_encoding:
                resp.encoding = resp.apparent_encoding

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Remove noise elements
            for tag in soup(['script', 'style', 'nav', 'header', 'footer',
                             'aside', 'iframe', 'figure', 'figcaption']):
                tag.decompose()

            # Try common article body selectors (ordered by specificity)
            content_selectors = [
                '#dic_area',                   # Naver news v2
                '#newsEndContents',            # Naver finance news
                '#articleBodyContents',         # Naver news legacy
                '#article-view-content-div',   # einfomax
                '.article-view-body',          # einfomax v2
                '.news_cnt_detail_wrap',       # MK
                '#article_body',               # Generic
                'article .content',            # Generic
                '.article_txt',                # Generic
                'article',                     # Broad fallback
            ]

            for selector in content_selectors:
                elem = soup.select_one(selector)
                if elem:
                    text = elem.get_text(separator='\n', strip=True)
                    if len(text) > 80:
                        return text[:max_chars]

            # Final fallback: join substantial <p> tags
            paragraphs = soup.find_all('p')
            texts = [p.get_text(strip=True) for p in paragraphs
                     if len(p.get_text(strip=True)) > 30]
            if texts:
                return '\n'.join(texts)[:max_chars]

            return ""
        except Exception as e:
            self.logger.debug(f"Failed to fetch article content: {url} ({e})")
            return ""

    def _collect_naver_news(self, stock_code: str, company_name: str) -> List[Dict]:
        """
        Collect news from Naver Finance.

        Args:
            stock_code: 6-digit stock code
            company_name: Company name

        Returns:
            List of news dictionaries with date, title, url, source
        """
        news_list = []
        url = f"https://finance.naver.com/item/news_news.naver?code={stock_code}"

        try:
            res = retry_request(
                requests.get,
                max_retries=3,
                base_delay=1,
                logger=self.logger,
                url=url,
                headers=self.headers,
                timeout=10
            )
            res.encoding = 'euc-kr'
            soup = BeautifulSoup(res.text, 'html.parser')

            # Find all news rows
            rows = soup.select('table.type2 tbody tr')

            for row in rows:
                try:
                    # Extract date and title
                    date_td = row.select_one('td.date')
                    title_td = row.select_one('td.title')

                    if not date_td or not title_td:
                        continue

                    date_text = date_td.text.strip()
                    title_elem = title_td.find('a')

                    if not title_elem:
                        continue

                    title = title_elem.text.strip()
                    news_url = title_elem.get('href', '')

                    # Parse date
                    try:
                        news_date = datetime.strptime(date_text, '%Y.%m.%d').date()
                    except ValueError:
                        # Skip if date parsing fails
                        continue

                    # Skip if before cutoff date
                    if news_date < self.cutoff_date.date():
                        continue

                    news_list.append({
                        'date': str(news_date),
                        'title': title,
                        'url': news_url,
                        'source': 'naver'
                    })

                except Exception as e:
                    self.logger.debug(f"Error parsing Naver news row: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error collecting Naver news: {e}")

        return news_list

    def _collect_google_news(self, company_name: str) -> List[Dict]:
        """
        Collect news from Google News RSS feed.

        Args:
            company_name: Company name (Korean)

        Returns:
            List of news dictionaries with date, title, url, source
        """
        news_list = []

        if feedparser is None:
            self.logger.warning("feedparser not installed - skipping Google News collection")
            return news_list

        try:
            # Google News RSS feed for the company
            rss_url = f"https://news.google.com/rss/search?q={company_name}+주식&hl=ko&gl=KR&ceid=KR:ko"

            feed = feedparser.parse(rss_url)

            if feed.bozo:
                self.logger.warning(f"RSS feed parsing issue: {feed.bozo_exception}")

            for entry in feed.entries[:50]:  # Limit to 50 entries
                try:
                    title = entry.get('title', '')
                    link = entry.get('link', '')
                    published = entry.get('published', '')

                    if not title or not link:
                        continue

                    # Parse published date
                    try:
                        # feedparser returns tuple, convert to datetime
                        from email.utils import parsedate_to_datetime
                        news_datetime = parsedate_to_datetime(published)
                        news_date = news_datetime.date()
                    except Exception:
                        # Fallback: try to parse common formats
                        try:
                            news_date = datetime.fromisoformat(published[:10]).date()
                        except Exception:
                            continue

                    # Skip if before cutoff date
                    if news_date < self.cutoff_date.date():
                        continue

                    news_list.append({
                        'date': str(news_date),
                        'title': title,
                        'url': link,
                        'source': 'google'
                    })

                except Exception as e:
                    self.logger.debug(f"Error parsing Google News entry: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error collecting Google News: {e}")

        return news_list

    def _collect_hankyung_consensus(self, stock_code: str, company_name: str) -> List[Dict]:
        """
        Collect analyst report listings from Hankyung Consensus.

        Extracts consensus data (broker reports, target prices) from
        the Nuxt.js SSR payload embedded in the page source.

        Args:
            stock_code: 6-digit stock code
            company_name: Company name

        Returns:
            List of news dictionaries with date, title, url, source
        """
        news_list = []
        url = f"https://markets.hankyung.com/stock/{stock_code}/consensus"

        try:
            with requests.Session() as session:
                session.mount("https://markets.hankyung.com", _WeakDHAdapter())
                res = retry_request(
                    session.get, max_retries=3, base_delay=1, logger=self.logger,
                    url=url, headers=self.headers, timeout=10
                )
            res.encoding = 'utf-8'
            text = res.text

            # Extract report data from Nuxt SSR payload
            titles = re.findall(r'"REPORT_TITLE"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            brokers = re.findall(r'"OFFICE_NAME"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            dates = re.findall(r'"REPORT_DATE"\s*:\s*"(\d{4}-\d{2}-\d{2})"', text)

            if titles and len(titles) == len(brokers) == len(dates):
                for title, broker, date_str in zip(titles, brokers, dates):
                    try:
                        report_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        if report_date < self.cutoff_date.date():
                            continue

                        news_list.append({
                            'date': date_str,
                            'title': f"[{broker}] {title}",
                            'url': url,
                            'source': 'hankyung_consensus'
                        })
                    except ValueError:
                        continue
            else:
                self.logger.debug(
                    f"Hankyung data count mismatch: "
                    f"titles={len(titles)}, brokers={len(brokers)}, dates={len(dates)}"
                )

        except Exception as e:
            self.logger.error(f"Error collecting Hankyung consensus: {e}")

        return news_list

    def _collect_einfomax_news(self, company_name: str) -> List[Dict]:
        """
        Collect financial news from Yonhap Infomax.

        Scrapes search results page for company-specific financial news.

        Args:
            company_name: Company name (Korean)

        Returns:
            List of news dictionaries with date, title, url, source
        """
        news_list = []
        url = (
            f"https://news.einfomax.co.kr/news/articleList.html"
            f"?sc_word={quote(company_name)}&sc_area=A&sc_order_by=E&view_type=sm"
        )

        try:
            res = retry_request(
                requests.get, max_retries=3, base_delay=1, logger=self.logger,
                url=url, headers=self.headers, timeout=10
            )
            res.encoding = 'utf-8'
            soup = BeautifulSoup(res.text, 'html.parser')

            # Parse article list items (ND Soft CMS structure)
            articles = soup.select('#section-list ul li')
            if not articles:
                articles = soup.select('.article-list-content ul li')

            for article in articles[:30]:
                try:
                    title_elem = article.select_one('h4 a, .titles a')
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '')
                    if link and not link.startswith('http'):
                        link = f"https://news.einfomax.co.kr{link}"

                    # Extract date from <em> tags
                    date_str = None
                    for em in article.select('em'):
                        text = em.get_text(strip=True)
                        match = re.match(r'(\d{4}\.\d{2}\.\d{2})', text)
                        if match:
                            date_str = match.group(1)
                            break

                    if not date_str:
                        continue

                    news_date = datetime.strptime(date_str, '%Y.%m.%d').date()
                    if news_date < self.cutoff_date.date():
                        continue

                    news_list.append({
                        'date': str(news_date),
                        'title': title,
                        'url': link,
                        'source': 'einfomax'
                    })

                except Exception as e:
                    self.logger.debug(f"Error parsing Einfomax article: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error collecting Einfomax news: {e}")

        return news_list

    def _collect_mk_news(self, company_name: str) -> List[Dict]:
        """
        Collect stock news from Maeil Business Newspaper (MK).

        Uses the MK Securities RSS feed and filters by company name.

        Args:
            company_name: Company name (Korean)

        Returns:
            List of news dictionaries with date, title, url, source
        """
        news_list = []

        if feedparser is None:
            self.logger.warning("feedparser not installed - skipping MK news collection")
            return news_list

        try:
            # MK Securities section RSS
            rss_url = "http://file.mk.co.kr/news/rss/rss_50200011.xml"
            feed = feedparser.parse(rss_url)

            if feed.bozo:
                self.logger.warning(f"MK RSS parsing issue: {feed.bozo_exception}")

            for entry in feed.entries:
                try:
                    title = entry.get('title', '')
                    link = entry.get('link', '')
                    published = entry.get('published', '')
                    description = entry.get('description', '') or entry.get('summary', '')

                    if not title or not link:
                        continue

                    # Filter: only include articles mentioning the company
                    if company_name not in title and company_name not in description:
                        continue

                    try:
                        from email.utils import parsedate_to_datetime
                        news_datetime = parsedate_to_datetime(published)
                        news_date = news_datetime.date()
                    except Exception:
                        continue

                    if news_date < self.cutoff_date.date():
                        continue

                    news_list.append({
                        'date': str(news_date),
                        'title': title,
                        'url': link,
                        'source': 'mk'
                    })

                except Exception as e:
                    self.logger.debug(f"Error parsing MK news entry: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error collecting MK news: {e}")

        return news_list

    def _collect_thebell_news(self, company_name: str) -> List[Dict]:
        """
        Collect financial news from TheBell (더벨).

        Scrapes search results from thebell.co.kr for company-specific
        capital markets and finance news.

        Args:
            company_name: Company name (Korean)

        Returns:
            List of news dictionaries with date, title, url, source
        """
        news_list = []
        url = (
            f"https://www.thebell.co.kr/search/search.asp"
            f"?keyword={quote(company_name)}&period=6MON&ord=NEWSDATE"
        )

        try:
            res = retry_request(
                requests.get, max_retries=3, base_delay=1, logger=self.logger,
                url=url, headers=self.headers, timeout=10
            )
            res.encoding = 'utf-8'
            soup = BeautifulSoup(res.text, 'html.parser')

            # Articles are in div.newsList.tp1 > ul > li
            articles = soup.select('div.newsList.tp1 ul li')

            for article in articles[:30]:
                try:
                    # Title and URL from dl > dt > a.txtE
                    title_elem = article.select_one('dl dt a.txtE')
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '')
                    if link and not link.startswith('http'):
                        link = f"https://www.thebell.co.kr{link}"

                    if not title or not link:
                        continue

                    # Date from dd.userBox > span.date (format: YYYY-MM-DD HH:MM:SS)
                    date_elem = article.select_one('dd.userBox span.date')
                    if not date_elem:
                        continue

                    date_text = date_elem.get_text(strip=True)
                    match = re.match(r'(\d{4}-\d{2}-\d{2})', date_text)
                    if not match:
                        continue
                    news_date = datetime.strptime(match.group(1), '%Y-%m-%d').date()

                    if news_date < self.cutoff_date.date():
                        continue

                    news_list.append({
                        'date': str(news_date),
                        'title': title,
                        'url': link,
                        'source': 'thebell'
                    })

                except Exception as e:
                    self.logger.debug(f"Error parsing TheBell article: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error collecting TheBell news: {e}")

        return news_list
