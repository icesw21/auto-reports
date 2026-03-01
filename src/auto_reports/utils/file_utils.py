"""
File utility functions for DART research report collector.

This module provides file-related helper functions including filename sanitization
and stock dictionary loading from JSON/CSV files.
"""

import json
import os
import re
from typing import Dict, Optional
from auto_reports.utils.logging import get_logger

logger = get_logger(__name__)

# Default stock list file
DEFAULT_STOCKS_JSON = "stocks.json"


def sanitize_filename(filename: str) -> str:
    r"""
    Remove forbidden characters from filename for Windows compatibility.

    Removes characters that are not allowed in Windows filenames:
    \ / : * ? " < > |

    Args:
        filename: Original filename string

    Returns:
        str: Sanitized filename with forbidden characters replaced by underscores

    Example:
        >>> sanitize_filename('Report: Q1 2024 <Draft>')
        'Report_ Q1 2024 _Draft_'
    """
    return re.sub(r'[\\/:*?"<>|]', '_', filename)


def load_stock_json(file_path: str = DEFAULT_STOCKS_JSON) -> Optional[Dict[str, str]]:
    """
    Load stock codes and company names from a JSON file.

    JSON format: {"종목코드": "회사명", ...}
    Example: {"005930": "삼성전자", "000660": "SK하이닉스"}

    Args:
        file_path: Path to JSON file (default: stocks.json)

    Returns:
        dict: Dictionary mapping stock codes to company names
        None: If file not found or parse error

    Example:
        >>> stock_dict = load_stock_json('stocks.json')
        >>> print(stock_dict)
        {'005930': '삼성전자', '000660': 'SK하이닉스'}
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"Stock JSON file not found: {file_path}")
            print(f"Error: {file_path} not found. Create it with format: {{\"종목코드\": \"회사명\"}}")
            return None

        logger.info(f"Reading stock list: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logger.error(f"Invalid JSON format: expected dict, got {type(data).__name__}")
            print(f'Error: {file_path} must be a JSON object {{"종목코드": "회사명"}}, not {type(data).__name__}')
            return None

        # Normalize codes to 6-digit zero-padded strings
        stock_dict = {}
        for code, name in data.items():
            clean_code = str(code).strip().zfill(6)
            stock_dict[clean_code] = str(name).strip()

        logger.info(f"Loaded {len(stock_dict)} stocks from {file_path}")
        return stock_dict

    except json.JSONDecodeError as e:
        logger.exception(f"JSON parse error in {file_path}")
        print(f"Error: Invalid JSON in {file_path}: {e}")
        return None
    except Exception as e:
        logger.exception("Error loading stock JSON")
        print(f"Error loading stock list: {e}")
        return None


def convert_csv_to_json(csv_path: str, json_path: str = DEFAULT_STOCKS_JSON) -> Optional[Dict[str, str]]:
    """
    Convert a CP949-encoded CSV stock list to JSON format.

    Args:
        csv_path: Path to CSV file (CP949 encoded)
        json_path: Output JSON file path (default: stocks.json)

    Returns:
        dict: Converted stock dictionary, or None on error
    """
    try:
        import pandas as pd

        logger.info(f"Converting CSV to JSON: {csv_path} -> {json_path}")
        df = pd.read_csv(csv_path, encoding='cp949')
        code_col = df.columns[2]
        name_col = df.columns[3]
        df = df.dropna(subset=[code_col, name_col])
        df[code_col] = df[code_col].astype(str).str.replace("'", "").str.strip().str.zfill(6)
        stock_dict = dict(zip(df[code_col], df[name_col]))

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stock_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"Converted {len(stock_dict)} stocks: {json_path}")
        print(f"Converted {len(stock_dict)} stocks from {csv_path} to {json_path}")
        return stock_dict

    except ImportError:
        logger.error("pandas required for CSV conversion: pip install pandas")
        print("Error: pandas is required for CSV conversion. Run: pip install pandas")
        return None
    except Exception as e:
        logger.exception("Error converting CSV to JSON")
        print(f"Error converting CSV: {e}")
        return None


