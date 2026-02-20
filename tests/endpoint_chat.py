"""
Interactive chat client for the running PIBot Serving API.

Usage:
    python tests/endpoint_chat.py
    python tests/endpoint_chat.py --url http://localhost:8000

Requirements:
- The API server must be running (e.g. `uvicorn app.main:app --reload`).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx


def _pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(", ", ": "))


def _format_head_inline(head: dict[str, Any] | None) -> str:
    if not head:
        return "null"
    return _compact_json(
        {
            "label": head.get("label"),
            "confidence": head.get("confidence"),
        }
    )


def _format_entities_normalized_expanded(value: Any) -> list[str]:
    """Render entities_normalized as expanded JSON block with inline lists."""
    if value is None:
        return ["null"]

    if not isinstance(value, dict):
        return [_compact_json(value)]

    items = list(value.items())
    if not items:
        return ["{}"]

    lines = ["{"]
    for idx, (key, item_value) in enumerate(items):
        suffix = "," if idx < len(items) - 1 else ""
        lines.append(f'      "{key}": {_compact_json(item_value)}{suffix}')
    lines.append("    }")
    return lines


def _print_readme_style_response(response_json: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("Response")
    print("=" * 80)

    routing = response_json.get("routing") or {}
    interpretation = response_json.get("interpretation") or {}
    intents = interpretation.get("intents") or {}
    entities_normalized_lines = _format_entities_normalized_expanded(
        interpretation.get("entities_normalized")
    )

    lines = [
        "{",
        f'  "text": {_compact_json(response_json.get("text"))},',
        '  "routing": {',
        f'    "macro": {_format_head_inline(routing.get("macro"))},',
        f'    "intent": {_format_head_inline(routing.get("intent"))},',
        f'    "context": {_format_head_inline(routing.get("context"))}',
        "  },",
        '  "interpretation": {',
        f'    "words": {_compact_json(interpretation.get("words") or [])},',
        '    "intents": {',
        f'      "calc_mode": {_format_head_inline(intents.get("calc_mode"))},',
        f'      "activity": {_format_head_inline(intents.get("activity"))},',
        f'      "region": {_format_head_inline(intents.get("region"))},',
        f'      "investment": {_format_head_inline(intents.get("investment"))},',
        f'      "req_form": {_format_head_inline(intents.get("req_form"))}',
        "    },",
        f'    "slot_tags": {_compact_json(interpretation.get("slot_tags") or [])},',
        f'    "entities": {_compact_json(interpretation.get("entities"))},',
        f'    "entities_normalized": {entities_normalized_lines[0]}',
        *entities_normalized_lines[1:],
        "  }",
        "}",
    ]

    print("\n".join(lines))
    print("=" * 80 + "\n")


def _check_server(client: httpx.Client, base_url: str) -> bool:
    try:
        resp = client.get(f"{base_url}/health", timeout=5.0)
    except Exception:
        print("No se pudo conectar al servidor.")
        print("Enciéndelo y vuelve a intentar:")
        print("  uvicorn app.main:app --reload")
        return False

    if resp.status_code != 200:
        print(f"Servidor respondió /health con status {resp.status_code}.")
        print("Verifica que esté encendido y sano antes de usar el chat.")
        return False

    health = resp.json()
    print("Servidor conectado:")
    print(_pretty_json(health))
    return True


def run_chat(base_url: str) -> int:
    print("PIBot endpoint chat interactivo")
    print("Comandos: /exit, /quit, /health")

    with httpx.Client(timeout=30.0) as client:
        if not _check_server(client, base_url):
            return 1

        while True:
            try:
                user_text = input("\nTú > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSaliendo...")
                return 0

            if not user_text:
                continue

            if user_text.lower() in {"/exit", "/quit"}:
                print("Saliendo...")
                return 0

            if user_text.lower() == "/health":
                _check_server(client, base_url)
                continue

            try:
                resp = client.post(
                    f"{base_url}/predict",
                    json={"text": user_text},
                )
            except Exception as exc:
                print(f"Error llamando al endpoint: {exc}")
                print("Asegúrate de que el server siga activo.")
                continue

            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}")
                try:
                    print(_pretty_json(resp.json()))
                except Exception:
                    print(resp.text)
                continue

            _print_readme_style_response(resp.json())


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive chat against PIBot /predict endpoint")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the running API server (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    return run_chat(base_url)


if __name__ == "__main__":
    sys.exit(main())
