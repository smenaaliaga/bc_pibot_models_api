"""
Prediction logic – tokenise, forward pass, decode head outputs + BIO slots.

Mirrors the logic in ``predict_cli.py`` from the training repo but is
structured as a stateless function that receives the loaded model bundle.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from app.model.loader import ModelBundle

logger = logging.getLogger(__name__)

# ── BIO entity extraction ────────────────────────────────────


def extract_entities_from_bio(
    words: List[str],
    bio_tags: List[str],
) -> Dict[str, List[str]]:
    """Convert per-word BIO tags into grouped entity spans."""
    entities: List[Tuple[str, str]] = []
    current_type: str | None = None
    current_tokens: List[str] = []

    for word, tag in zip(words, bio_tags):
        if not tag or tag == "O":
            if current_type and current_tokens:
                entities.append((current_type, " ".join(current_tokens)))
            current_type = None
            current_tokens = []
            continue

        if "-" not in tag:
            if current_type and current_tokens:
                entities.append((current_type, " ".join(current_tokens)))
            current_type = None
            current_tokens = []
            continue

        prefix, entity_type = tag.split("-", 1)

        if prefix == "B":
            if current_type and current_tokens:
                entities.append((current_type, " ".join(current_tokens)))
            current_type = entity_type
            current_tokens = [word]
        elif prefix == "I" and current_type == entity_type:
            current_tokens.append(word)
        else:
            if current_type and current_tokens:
                entities.append((current_type, " ".join(current_tokens)))
            current_type = entity_type
            current_tokens = [word]

    if current_type and current_tokens:
        entities.append((current_type, " ".join(current_tokens)))

    grouped: Dict[str, List[str]] = defaultdict(list)
    for etype, text in entities:
        grouped[etype].append(text)
    return dict(grouped)


# ── Head decoder helper ──────────────────────────────────────


def _decode_head(
    logits: torch.Tensor,
    label_list: List[str],
) -> Tuple[str, float]:
    """Return (predicted_label, confidence) for a single classification head."""
    probs = torch.softmax(logits, dim=-1)
    pred_idx = probs.argmax(dim=-1).item()
    label = label_list[pred_idx] if pred_idx < len(label_list) else "UNK"
    confidence = probs[0][pred_idx].item()
    return label, round(confidence, 6)


def _project_slot_predictions_to_words(
    *,
    words: List[str],
    word_ids: List[int | None],
    attention_mask: List[int],
    slot_pred_ids: List[int],
    slot_label_lst: List[str],
) -> List[str]:
    """Project token-level slot IDs to one BIO tag per original word.

    Handles CRF implementations that decode either full token sequences
    (including special tokens) or only word-piece positions.
    """
    slot_tags_per_word = ["O"] * len(words)
    if not words:
        return slot_tags_per_word

    seq_len = int(sum(attention_mask))
    token_positions = list(range(seq_len))
    word_token_positions = [i for i in token_positions if i < len(word_ids) and word_ids[i] is not None]

    if len(slot_pred_ids) == len(token_positions):
        pred_by_token_pos = {pos: slot_pred_ids[pos] for pos in token_positions}
    elif len(slot_pred_ids) == len(word_token_positions):
        pred_by_token_pos = {
            pos: slot_pred_ids[idx]
            for idx, pos in enumerate(word_token_positions)
        }
    else:
        # Fallback: align from the left over the valid attention span.
        pred_by_token_pos = {
            pos: slot_pred_ids[idx]
            for idx, pos in enumerate(token_positions[: len(slot_pred_ids)])
        }

    seen_word_idxs: set[int] = set()
    for token_pos in token_positions:
        if token_pos >= len(word_ids):
            break

        word_idx = word_ids[token_pos]
        if word_idx is None or word_idx in seen_word_idxs:
            continue

        seen_word_idxs.add(word_idx)
        pred_id = pred_by_token_pos.get(token_pos)
        if pred_id is None:
            continue
        if 0 <= pred_id < len(slot_label_lst):
            slot_tags_per_word[word_idx] = slot_label_lst[pred_id]

    return slot_tags_per_word


# ── Main predict function ────────────────────────────────────


def predict(bundle: ModelBundle, text: str) -> dict:
    """
    Run inference on a single text and return structured results.

    Returns
    -------
    dict with keys:
        text, words,
        calc_mode, calc_mode_confidence,
        activity, activity_confidence,
        region, region_confidence,
        investment, investment_confidence,
        req_form, req_form_confidence,
        slot_tags, entities
    """
    words = text.split()
    if not words:
        return {"text": text, "words": [], "entities": {}, "slot_tags": []}

    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device
    args = bundle.train_args
    max_seq_len = getattr(args, "max_seq_len", 64)

    # ── Tokenize preserving word alignment (works for BERT and DeBERTa) ──
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        return_tensors="pt",
    )

    raw_word_ids = encoded.word_ids(batch_index=0) if hasattr(encoded, "word_ids") else None
    if raw_word_ids is None:
        raise RuntimeError("Tokenizer does not provide word alignment (word_ids).")

    word_ids: List[int | None] = [wid if wid is not None else None for wid in raw_word_ids]

    input_ids_t = encoded["input_ids"].to(device)
    attention_mask_t = encoded["attention_mask"].to(device)
    token_type_ids_t = encoded.get("token_type_ids")
    if token_type_ids_t is not None:
        token_type_ids_t = token_type_ids_t.to(device)

    # ── Forward pass ──────────────────────────────────────────
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            token_type_ids=token_type_ids_t,
            calc_mode_label_ids=None,
            activity_label_ids=None,
            region_label_ids=None,
            investment_label_ids=None,
            req_form_label_ids=None,
            slot_labels_ids=None,
        )

    logits_tuple = outputs[1]  # (calc, activity, region, investment, req_form, slot)
    (
        calc_mode_logits,
        activity_logits,
        region_logits,
        investment_logits,
        req_form_logits,
        slot_logits,
    ) = logits_tuple

    # ── Decode classification heads ───────────────────────────
    calc_mode_label, calc_mode_conf = _decode_head(calc_mode_logits, bundle.labels["calc_mode"])
    activity_label, activity_conf = _decode_head(activity_logits, bundle.labels["activity"])
    region_label, region_conf = _decode_head(region_logits, bundle.labels["region"])
    investment_label, investment_conf = _decode_head(investment_logits, bundle.labels["investment"])
    req_form_label, req_form_conf = _decode_head(req_form_logits, bundle.labels["req_form"])

    # ── Decode slot (NER) ─────────────────────────────────────
    use_crf = getattr(args, "use_crf", False)
    slot_label_lst = bundle.labels["slot"]

    if use_crf and hasattr(model, "crf"):
        if hasattr(model.crf, "decode"):
            slot_pred_ids = model.crf.decode(slot_logits, mask=attention_mask_t.bool())[0]
        else:
            slot_pred_ids = model.crf.viterbi_decode(slot_logits, attention_mask_t.bool())[0]
    else:
        slot_pred_ids = torch.argmax(slot_logits, dim=-1).squeeze(0).tolist()

    slot_tags_per_word = _project_slot_predictions_to_words(
        words=words,
        word_ids=word_ids,
        attention_mask=attention_mask_t.squeeze(0).tolist(),
        slot_pred_ids=list(slot_pred_ids),
        slot_label_lst=slot_label_lst,
    )
    entities = extract_entities_from_bio(words, slot_tags_per_word)

    return {
        "text": text,
        "words": words,
        "calc_mode": calc_mode_label,
        "calc_mode_confidence": calc_mode_conf,
        "activity": activity_label,
        "activity_confidence": activity_conf,
        "region": region_label,
        "region_confidence": region_conf,
        "investment": investment_label,
        "investment_confidence": investment_conf,
        "req_form": req_form_label,
        "req_form_confidence": req_form_conf,
        "slot_tags": slot_tags_per_word,
        "entities": entities,
    }
