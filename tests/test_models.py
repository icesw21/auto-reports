"""Tests for Pydantic data models."""

import json
from pathlib import Path

import pytest

from auto_reports.models.disclosure import (
    CBIssuance,
    Contract,
    ConvertExercise,
    ConvertPriceChange,
    Performance,
    parse_korean_number,
    parse_korean_float,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestParseKoreanNumber:
    def test_with_commas(self):
        assert parse_korean_number("1,932,000,000") == 1_932_000_000

    def test_with_won_suffix(self):
        assert parse_korean_number("1,932,000,000원") == 1_932_000_000

    def test_simple(self):
        assert parse_korean_number("5,370") == 5370

    def test_negative(self):
        assert parse_korean_number("-6,372,865,544") == -6_372_865_544

    def test_dash_returns_none(self):
        assert parse_korean_number("-") is None

    def test_none_returns_none(self):
        assert parse_korean_number(None) is None

    def test_empty_string_returns_none(self):
        assert parse_korean_number("") is None

    def test_int_passthrough(self):
        assert parse_korean_number(10_000_000_000) == 10_000_000_000

    def test_float_truncated(self):
        assert parse_korean_number(123.7) == 123

    def test_percentage_strip(self):
        assert parse_korean_number("5.44%") == 5

    def test_shares_suffix(self):
        assert parse_korean_number("522,279주") == 522_279


class TestParseKoreanFloat:
    def test_percentage(self):
        assert parse_korean_float("5.44%") == pytest.approx(5.44)

    def test_negative_float(self):
        assert parse_korean_float("-3.2") == pytest.approx(-3.2)


class TestConvertExercise:
    def test_from_json(self):
        with open(FIXTURES / "convert.json", encoding="utf-8") as f:
            data = json.load(f)
        model = ConvertExercise(**data)
        assert model.cumulative_shares_int == 359_766
        assert len(model.daily_claims) == 1
        assert len(model.cb_balance) == 2
        assert model.cb_balance[0].series == 2
        assert model.cb_balance[1].conversion_price_int == 13_844


class TestConvertPriceChange:
    def test_from_json(self):
        with open(FIXTURES / "convert-price-change.json", encoding="utf-8") as f:
            data = json.load(f)
        model = ConvertPriceChange(**data)
        assert model.adjustment.price_before_int == 13_844
        assert model.adjustment.price_after_int == 9_691
        assert model.share_change.shares_before_int == 613_984
        assert model.share_change.shares_after_int == 877_102


class TestContract:
    def test_from_json(self):
        with open(FIXTURES / "contract.json", encoding="utf-8") as f:
            data = json.load(f)
        inner_key = list(data.keys())[0]
        model = Contract(contract_type_name=inner_key, **data[inner_key])
        assert model.details.total_amount_int == 20_319_210_726
        assert len(model.notes) == 6
        assert model.region == "해외"


class TestPerformance:
    def test_from_json(self):
        with open(FIXTURES / "performance.json", encoding="utf-8") as f:
            data = json.load(f)
        model = Performance(**data)
        assert model.statement_type == "연결"
        assert model.revenue.current_int == 95_639_307_299
        assert model.net_income.turnaround == "흑자전환"


class TestCBIssuance:
    def test_from_json(self):
        with open(FIXTURES / "issue.json", encoding="utf-8") as f:
            data = json.load(f)
        inner_key = list(data.keys())[0]
        model = CBIssuance(issuance_type_name=inner_key, **data[inner_key])
        assert model.face_value == 10_000_000_000
        assert model.conversion_terms.conversion_price == 13_956
        assert model.conversion_terms.share_count == 716_537
