from datetime import datetime

from app.model.normalizer import (
    normalize_activity,
    normalize_period,
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
    month_start = f"{now.year:04d}-{now.month:02d}-01"
    if now.month == 12:
        next_month = datetime(now.year + 1, 1, 1)
    else:
        next_month = datetime(now.year, now.month + 1, 1)
    last_day = (next_month - datetime.resolution).day
    month_end = f"{now.year:04d}-{now.month:02d}-{last_day:02d}"
    assert result["period"] == [month_start, month_end]


def test_period_range_returns_tuple_string():
    result = normalize_entities(
        entities={"period": ["de febrero a marzo del 2024"], "indicator": ["imacec"]},
        calc_mode="original",
        req_form="range",
    )

    assert result["period"] == ["2024-02-01", "2024-03-31"]


def test_period_range_with_two_years_uses_correct_bounds():
    result = normalize_entities(
        entities={"period": ["entre marzo del 2023 y enero del 2024"], "indicator": ["imacec"]},
        calc_mode="original",
        req_form="range",
    )

    assert result["period"] == ["2023-03-01", "2024-01-31"]


def test_period_range_disordered_same_year_is_sorted_ascending():
    result = normalize_entities(
        entities={"period": ["entre mayo del 2023 y enero del 2023"], "indicator": ["imacec"]},
        calc_mode="original",
        req_form="range",
    )

    assert result["period"] == ["2023-01-01", "2023-05-31"]


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


def test_period_without_year_assumes_current_year():
    normalized, failed = normalize_period("enero")
    current_year = datetime.now().year

    assert normalized == f"{current_year:04d}-01-01"
    assert failed == []


def test_period_range_without_year_assumes_current_year():
    result = normalize_entities(
        entities={"period": ["entre enero y mayo"], "indicator": ["imacec"]},
        calc_mode="original",
        req_form="range",
    )
    current_year = datetime.now().year

    assert result["period"] == [f"{current_year:04d}-01-01", f"{current_year:04d}-05-31"]


def test_activity_conjunction_is_split_and_normalized():
    result = normalize_entities(
        entities={"activity": ["mineria y no mineria"], "period": ["mayo"]},
        calc_mode="prev_period",
        req_form="range",
    )

    assert result["activity"] == ["mineria", "no_mineria"]


def test_indicator_keeps_pib_regional_and_discards_banco_central():
    result = normalize_entities(
        entities={
            "indicator": ["banco central", "pib regional"],
            "region": ["chile"],
            "period": ["durante el año 2024"],
        },
        calc_mode="original",
        req_form="range",
    )

    assert result["indicator"] == ["pib"]


def test_pib_without_frequency_infers_quarterly_frequency():
    result = normalize_entities(
        entities={
            "indicator": ["banco central", "pib regional"],
            "region": ["chile"],
            "period": ["durante el año 2024"],
        },
        calc_mode="original",
        req_form="range",
    )

    assert result["indicator"] == ["pib"]
    assert result["frequency"] == ["q"]


def test_pib_activity_conjunction_splits_and_keeps_both_entities():
    result = normalize_entities(
        entities={
            "indicator": ["pib"],
            "activity": ["minero y comercial"],
            "frequency": ["anual"],
            "seasonality": ["con estacionalidad"],
        },
        calc_mode="prev_period",
        req_form="latest",
    )

    assert result["activity"] == ["mineria", "comercio"]


def test_explicit_seasonality_overrides_prev_period_default():
    result = normalize_entities(
        entities={
            "indicator": ["pib"],
            "seasonality": ["sin ajuste estacional"],
        },
        calc_mode="prev_period",
        req_form="point",
    )

    assert result["seasonality"] == ["nsa"]


def test_explicit_seasonality_overrides_yoy_default():
    result = normalize_entities(
        entities={
            "indicator": ["pib"],
            "seasonality": ["desestacionalizado"],
        },
        calc_mode="yoy",
        req_form="point",
    )

    assert result["seasonality"] == ["sa"]


def test_quarter_period_range_text_returns_quarter_boundaries():
    result = normalize_entities(
        entities={
            "indicator": ["pib"],
            "period": ["del primer trimestre al cuarto trimestre del 2024"],
        },
        calc_mode="original",
        req_form="range",
    )

    assert result["period"] == ["2024-01-01", "2024-12-31"]


def test_frequency_q_snaps_period_to_quarter_start():
    result = normalize_entities(
        entities={
            "indicator": ["pib"],
            "frequency": ["trimestral"],
            "period": ["febrero 2024"],
        },
        calc_mode="original",
        req_form="point",
    )

    assert result["frequency"] == ["q"]
    assert result["period"] == ["2024-01-01", "2024-03-31"]


def test_quarter_range_uses_last_day_on_upper_bound():
    result = normalize_entities(
        entities={
            "indicator": ["pib"],
            "period": ["entre el primer trimestre y el tercer trimestre del 2024"],
        },
        calc_mode="original",
        req_form="range",
    )

    assert result["period"] == ["2024-01-01", "2024-09-30"]


def test_year_only_range_uses_last_day_of_year_on_upper_bound():
    result = normalize_entities(
        entities={
            "indicator": ["imacec"],
            "period": ["durante el 2024"],
        },
        calc_mode="yoy",
        req_form="range",
    )

    assert result["period"] == ["2024-01-01", "2024-12-31"]


def test_explicit_imacec_without_frequency_infers_monthly_frequency():
    result = normalize_entities(
        entities={
            "indicator": ["imacec"],
            "period": ["durante el 2024"],
        },
        calc_mode="yoy",
        req_form="range",
    )

    assert result["frequency"] == ["m"]


def test_generic_indicator_with_region_intent_and_entity_infers_pib_q():
    result = normalize_entities(
        entities={
            "indicator": ["economia"],
            "region": ["chile"],
        },
        calc_mode="original",
        req_form="point",
        intents={"region": "specific", "investment": "none"},
    )

    assert result["indicator"] == ["pib"]
    assert result["frequency"] == ["q"]


def test_generic_indicator_with_investment_intent_and_entity_infers_pib_q():
    result = normalize_entities(
        entities={
            "indicator": ["economia"],
            "investment": ["inversion"],
        },
        calc_mode="original",
        req_form="point",
        intents={"region": "none", "investment": "specific"},
    )

    assert result["indicator"] == ["pib"]
    assert result["frequency"] == ["q"]


def test_generic_indicator_with_region_intent_without_entity_still_infers_pib_q():
    result = normalize_entities(
        entities={
            "indicator": ["economia"],
        },
        calc_mode="original",
        req_form="point",
        intents={"region": "specific", "investment": "none"},
    )

    assert result["indicator"] == ["pib"]
    assert result["frequency"] == ["q"]


def test_pib_point_with_year_only_period_forces_annual_frequency():
    result = normalize_entities(
        entities={
            "activity": ["mineria"],
            "period": ["año 2023"],
        },
        calc_mode="contribution",
        req_form="point",
        intents={"region": "general", "investment": "none"},
    )

    assert result["indicator"] == ["pib"]
    assert result["frequency"] == ["a"]
    assert result["period"] == ["2023-01-01", "2023-12-31"]
