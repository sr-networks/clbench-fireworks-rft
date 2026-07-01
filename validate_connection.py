#!/usr/bin/env python3
"""Step 0 — connectivity smoke test.

Confirms that:
  1. FIREWORKS_API_KEY is set and valid.
  2. The chosen Qwen3 base model is reachable for inference via LiteLLM
     (the same client eval-protocol uses under the hood).
  3. Structured JSON output works — the mechanism that replaces the
     vLLM `guided_json` we relied on in the verifiers/Prime setup.

Run BEFORE building the MCP gym, so we know the pipeline + auth + model id
are good in isolation.

    python fireworks_rft/validate_connection.py
"""

from __future__ import annotations

import json
import os
import sys

# Fireworks model id for the free-tier (<16B) Qwen3-8B base model.
# Confirm the exact id once authed with:  firectl list models | grep -i qwen3
MODEL = "fireworks_ai/accounts/fireworks/models/qwen3-8b"

POKER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "PokerAction",
        "schema": {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "action": {"type": "string", "enum": ["FOLD", "CALL", "CHECK", "RAISE"]},
                "amount": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            },
            "required": ["thinking", "action"],
        },
    },
}


def main() -> int:
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("FIREWORKS_API_KEY is not set. See fireworks_rft/README.md, Step 1.")
        return 2

    try:
        import litellm
    except ImportError:
        print("litellm missing — it ships with eval-protocol. `pip install eval-protocol`.")
        return 2

    prompt = (
        "You are heads-up in Texas Hold'em. You hold As Ks. Board is empty (preflop). "
        "Opponent (a calling station) limped. Pot is 3 BB. Legal actions: CHECK, RAISE, FOLD. "
        "Respond with a single JSON object matching the schema."
    )

    print(f"Calling {MODEL} with structured output...")
    resp = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=POKER_SCHEMA,
        temperature=0.7,
        max_tokens=512,
    )
    content = resp.choices[0].message.content
    print("\nRaw response:\n" + content)

    parsed = json.loads(content)  # raises if structured output failed
    assert parsed["action"] in {"FOLD", "CALL", "CHECK", "RAISE"}, parsed
    print(f"\nOK — model reachable, structured output valid. action={parsed['action']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
