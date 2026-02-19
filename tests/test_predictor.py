"""
Unit tests for the BIO entity extraction logic.
These run without any model artefacts.
"""

from app.model.predictor import extract_entities_from_bio


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
