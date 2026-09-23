"""parse.py - read tool calls out of a Qwen3 completion. Stdlib only.

Qwen3's chat template renders an assistant tool call as

    <tool_call>
    {"name": "intent__HassTurnOn", "arguments": {"name": "Back Hall Lamp"}}
    </tool_call>

and with thinking off it prefills an empty think block first:
`<think>\n\n</think>\n\n`. So a completion looks like

    <think>

    </think>

    <tool_call>
    {"name": "todo__HassListAddItem", "arguments": {"name": "Shed Jobs", "item": "wood glue"}}
    </tool_call>

format: 1 when there is at least one well formed block and no malformed block,
else 0. Well formed = the block's text parses as a json object, `name` is a non
empty string, and `arguments`, if present, is an object. Everything outside the
blocks (the think block, prose, an apology) is ignored.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Non greedy so two calls in one completion are two blocks, in order.
TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

OPEN_TAG = "<tool_call>"


def parse_tool_calls(text: str | None) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Return (calls, format_score, notes).

    calls is a list of {"name": str, "arguments": dict} in the order they appear.
    format_score is 1 or 0. notes says what was wrong, one line per problem, for
    the log.
    """
    text = text or ""
    notes: list[str] = []
    calls: list[dict[str, Any]] = []
    malformed = 0

    blocks = TOOL_CALL_BLOCK.findall(text)
    # An opened block that never closes cannot be found by the regex, so count
    # the opening tags: more openings than blocks means one was left hanging.
    unclosed = text.count(OPEN_TAG) - len(blocks)
    if unclosed > 0:
        malformed += unclosed
        notes.append(f"{unclosed} unclosed <tool_call> block")

    for block in blocks:
        body = block.strip()
        try:
            payload = json.loads(body)
        except ValueError as err:  # json.JSONDecodeError is a ValueError
            malformed += 1
            notes.append(f"block is not json: {err}")
            continue
        if not isinstance(payload, dict):
            malformed += 1
            notes.append(f"block is json but not an object: {type(payload).__name__}")
            continue
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            malformed += 1
            notes.append(f"block has no tool name: {body[:80]!r}")
            continue
        args = payload.get("arguments", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            malformed += 1
            notes.append(f"{name}: arguments is not an object: {type(args).__name__}")
            continue
        calls.append({"name": name, "arguments": args})

    if not calls and not malformed:
        notes.append("no tool call in the completion")
    format_score = 1 if calls and malformed == 0 else 0
    return calls, format_score, notes


# ---------------------------------------------------------------------------
# Home Assistant's own argument repair, used only by serve.py --repair-args.
# Verbatim from homeassistant/components/ollama/entity.py (HA 2026.9.3).
# ---------------------------------------------------------------------------


def _fix_invalid_arguments(value: Any) -> Any:
    """Attempt to repair incorrectly formatted json function arguments.

    Small models (for example llama3.1 8B) may produce invalid argument values
    which we attempt to repair here.
    """
    if not isinstance(value, str):
        return value
    if (value.startswith("[") and value.endswith("]")) or (
        value.startswith("{") and value.endswith("}")
    ):
        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            pass
    return value


def parse_tool_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ollama tool arguments.

    This function improves tool use quality by fixing common mistakes made by
    small local tool use models. This will repair invalid json arguments and
    omit unnecessary arguments with empty values that will fail intent parsing.
    """
    return {
        k: _fix_invalid_arguments(v)
        for k, v in arguments.items()
        if v is not None and v != ""
    }

