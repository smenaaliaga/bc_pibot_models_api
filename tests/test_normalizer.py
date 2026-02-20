from datetime import datetime

from app.model.normalizer import (
    normalize_activity,
    normalize_entities,
    normalize_frequency,
    normalize_investment,
    normalize_region,
    normalize_seasonality,
)


def test_period_latest_fallback_to_today_when_missing_period_entity():
    result = normalize_entities(
        entities={"indicator": ["imacec"]},
        calc_mode="original",
        req_form="latest",
    )

    now = datetime.now()
    expected_today = f"01-{now.month:02d}-{now.year}"
    assert result["period"] == expected_today


def test_period_range_returns_tuple_string():
    result = normalize_entities(
        entities={"period": ["de febrero a marzo del 2024"], "indicator": ["imacec"]},
        calc_mode="original",
        req_form="range",
    )

    assert result["period"] == ["01-02-2024", "01-03-2024"]


def test_period_range_with_two_years_uses_correct_bounds():
    result = normalize_entities(
        entities={"period": ["entre marzo del 2023 y enero del 2024"], "indicator": ["imacec"]},
        calc_mode="original",
        req_form="range",
    )

    assert result["period"] == ["01-03-2023", "01-01-2024"]


def test_period_range_disordered_same_year_is_sorted_ascending():
    result = normalize_entities(
        entities={"period": ["entre mayo del 2023 y enero del 2023"], "indicator": ["imacec"]},
        calc_mode="original",
        req_form="range",
    )

    assert result["period"] == ["01-01-2023", "01-05-2023"]


def test_activity_negative_phrase_prefers_no_mineria():
    activity, failed = normalize_activity("no mineri", "imacec")

    assert activity == "no_mineria"
    assert failed == []


def test_seasonality_negative_phrase_maps_to_nsa():
    assert normalize_seasonality("sin ajuste estacional", "original") == "nsa"


def test_frequency_best_match_trimestral_typo():
    assert normalize_frequency("trimesral") == "q"


def test_region_and_investment_best_match_typos():
    region, failed_region = normalize_region("metropolitan")
    investment, failed_investment = normalize_investment("invercion")

    assert region == "metropolitana"
    assert failed_region == []
    assert investment == "inversion"
    assert failed_investment == []


def test_generic_indicator_without_frequency_defaults_to_imacec_and_m():
    result = normalize_entities(
        entities={"indicator": ["economia"]},
        calc_mode="original",
        req_form="point",
    )

    assert result["indicator"] == ["imacec"]
    assert result["frequency"] == ["m"]


def test_generic_indicator_with_quarterly_frequency_defaults_to_pib():
    result = normalize_entities(
        entities={"indicator": ["economia"], "frequency": ["trimestral"]},
        calc_mode="original",
        req_form="point",
    )

    assert result["frequency"] == ["q"]
    assert result["indicator"] == ["pib"]
