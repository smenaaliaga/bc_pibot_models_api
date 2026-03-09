"""
Unit tests for the BIO entity extraction logic.
These run without any model artefacts.
"""

from app.model.predictor import extract_entities_from_bio
from app.model.predictor import _project_slot_predictions_to_words
from app.model.predictor import _apply_slot_fallback_heuristics


def test_simple_bio():
    words = ["el", "imacec", "de", "junio"]
    tags = ["O", "B-indicator", "O", "B-period"]
    result = extract_entities_from_bio(words, tags)
    assert result == {"indicator": ["imacec"], "period": ["junio"]}


def test_multi_token_entity():
    words = ["producto", "interno", "bruto", "nominal"]
    tags = ["B-indicator", "I-indicator", "I-indicator", "O"]
    result = extract_entities_from_bio(words, tags)
    assert result == {"indicator": ["producto interno bruto"]}


def test_all_o():
    words = ["hola", "mundo"]
    tags = ["O", "O"]
    result = extract_entities_from_bio(words, tags)
    assert result == {}


def test_empty():
    assert extract_entities_from_bio([], []) == {}


def test_consecutive_b_tags():
    words = ["imacec", "pib"]
    tags = ["B-indicator", "B-indicator"]
    result = extract_entities_from_bio(words, tags)
    assert result == {"indicator": ["imacec", "pib"]}


def test_different_entity_types():
    words = ["imacec", "de", "junio", "2024"]
    tags = ["B-indicator", "O", "B-period", "I-period"]
    result = extract_entities_from_bio(words, tags)
    assert result == {"indicator": ["imacec"], "period": ["junio 2024"]}


def test_project_slot_predictions_with_special_tokens():
    words = ["imacec", "junio"]
    word_ids = [None, 0, 1, None, None, None]
    attention_mask = [1, 1, 1, 1, 0, 0]
    slot_pred_ids = [0, 1, 2, 0]
    slot_labels = ["O", "B-indicator", "B-period"]

    projected = _project_slot_predictions_to_words(
        words=words,
        word_ids=word_ids,
        attention_mask=attention_mask,
        slot_pred_ids=slot_pred_ids,
        slot_label_lst=slot_labels,
    )

    assert projected == ["B-indicator", "B-period"]


def test_project_slot_predictions_word_positions_only():
    words = ["no", "minerio", "2025"]
    word_ids = [None, 0, 1, 2, None, None]
    attention_mask = [1, 1, 1, 1, 1, 0]
    slot_pred_ids = [1, 2, 3]
    slot_labels = ["O", "B-activity", "I-activity", "B-period"]

    projected = _project_slot_predictions_to_words(
        words=words,
        word_ids=word_ids,
        attention_mask=attention_mask,
        slot_pred_ids=slot_pred_ids,
        slot_label_lst=slot_labels,
    )

    assert projected == ["B-activity", "I-activity", "B-period"]


def test_slot_fallback_marks_explicit_year_as_period():
    words = ["cuanto", "crecio", "en", "1974"]
    tags = ["O", "O", "O", "O"]

    result = _apply_slot_fallback_heuristics(words, tags)
    assert result == ["O", "O", "O", "B-period"]


def test_slot_fallback_marks_month_when_year_present():
    words = ["imacec", "de", "mayo", "2025"]
    tags = ["O", "O", "O", "O"]

    result = _apply_slot_fallback_heuristics(words, tags)
    assert result == ["O", "O", "B-period", "I-period"]
