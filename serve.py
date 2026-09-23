#!/usr/bin/env python
"""serve.py - an OpenAI-compatible chat-completions server for hua-1.7b.

A small server, stdlib http, one model on the GPU in bf16, requests served one at a
time behind a lock. It is written for one client, Home Assistant's core `llama_cpp`
integration, which sends

    client.chat.completions.create(messages=..., model=..., tools=...,
        max_tokens=3000, top_p=1.0, temperature=0.7, user=..., stream=False)

    tools are OpenAI function specs:
        {"type": "function", "function": {"name", "parameters", "description"}}

    the system message is Home Assistant's Assist prompt, which lists the home's
    devices

    after a tool call it runs the tool and comes back with role="tool" messages, up to
    10 turns, and the last text reply is what the user hears

The prompt: the incoming messages AND the incoming tools go through Qwen3's own chat
template with `enable_thinking=False, add_generation_prompt=True`, which is how the
model was trained and scored. The `<tool_call>` blocks in the answer are parsed back
out (parse.py) and returned as OpenAI `tool_calls`.

Endpoints:
    POST /v1/chat/completions   non-streaming, and SSE when the client asks
    GET  /v1/models             the integration lists models when the entry loads
    GET  /health                model, requests served, mean ms

Run it:
    python serve.py                                   # neelabhbuilds/hua-1.7b from Hugging Face
    python serve.py --model-path /path/to/hua-1.7b    # a local copy of the weights

The `--model-name` is the id this server answers to (default hua-1.7b). A request
whose `model` field does not match is refused with 404, so a client pointed at the
wrong server fails loudly instead of being answered by other weights.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from parse import TOOL_CALL_BLOCK, parse_tool_args, parse_tool_calls

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

DEFAULT_MODEL_PATH = "neelabhbuilds/hua-1.7b"
DEFAULT_MODEL_NAME = "hua-1.7b"


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------


def to_template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI chat messages -> what Qwen3's chat template wants.

    Home Assistant sends four shapes (components/llama_cpp/entity.py
    _convert_content_to_chat_message): a system message, a user message, an
    assistant message that may carry `tool_calls` whose `arguments` is a json
    STRING, and a `role: "tool"` message carrying the tool result and its
    tool_call_id. Qwen3's template reads `message.tool_calls[].function.name` and
    `.arguments` (string or object, both handled by the template) and wraps a run of
    tool messages in a single `<tool_response>` user turn.

    A user message can also be a list of content parts when an attachment is
    involved; only the text parts are kept.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role") or "user"
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text") or ""
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        content = content or ""
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        pass  # the template renders a string verbatim
                calls.append(
                    {
                        "type": "function",
                        "function": {
                            "name": function.get("name") or "",
                            "arguments": {} if arguments is None else arguments,
                        },
                    }
                )
            out.append({"role": "assistant", "content": content, "tool_calls": calls})
        else:
            out.append({"role": role, "content": content})
    return out


def render_chat(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
    """The exact text the model sees. The one place the prompt is built."""
    return tokenizer.apply_chat_template(
        to_template_messages(messages),
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class Model:
    """The model in bf16 on the GPU, with SDPA attention and left padding."""

    def __init__(self, model_path: str, device: str) -> None:
        import torch  # noqa: PLC0415 - only the server needs it
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.torch = torch
        self.device = device
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        log(f"loading {model_path} in bf16 with SDPA attention on {device}")
        started = time.monotonic()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
        except TypeError:  # transformers 4.x spelled it torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
            )
        self.model.to(device)
        self.model.eval()
        stop = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        configured = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(configured, int):
            stop.add(configured)
        elif isinstance(configured, (list, tuple)):
            stop.update(int(token) for token in configured)
        self.stop_ids = {token for token in stop if token is not None}
        log(f"loaded in {time.monotonic() - started:.1f} s; stop ids {sorted(self.stop_ids)}")

    def peak_gib(self) -> float:
        if not self.torch.cuda.is_available():
            return 0.0
        return self.torch.cuda.max_memory_allocated() / 2**30

    def generate(
        self,
        text: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> tuple[str, int, int, bool]:
        """Return (completion text, prompt tokens, new tokens, hit the cap)."""
        torch = self.torch
        batch = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        prompt_tokens = int(batch["input_ids"].shape[1])
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**batch, **kwargs)
        ids = out[0, prompt_tokens:].tolist()
        kept, truncated = self._trim(ids, max_new_tokens)
        # skip_special_tokens=True, as in training; Qwen3 marks <tool_call> as an
        # ordinary token, so the call survives.
        return self.tokenizer.decode(kept, skip_special_tokens=True), prompt_tokens, len(kept), truncated

    def _trim(self, ids: list[int], cap: int) -> tuple[list[int], bool]:
        for position, token in enumerate(ids):
            if token in self.stop_ids:
                return ids[:position], False
        return ids, len(ids) >= cap


# ---------------------------------------------------------------------------
# the OpenAI shapes
# ---------------------------------------------------------------------------


def split_completion(text: str) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Completion text -> (visible content, tool calls, parse notes).

    The think block is empty (thinking is off) and the tool call blocks are the
    structured part, so neither belongs in `content`: a real OpenAI-compatible
    server hands back the prose only. Home Assistant would otherwise log the raw
    `<tool_call>` json as the assistant's words and the eval would score that text.
    """
    calls, _format_score, notes = parse_tool_calls(text)
    content = THINK_BLOCK.sub("", text)
    content = TOOL_CALL_BLOCK.sub("", content)
    return content.strip(), calls, notes


def tool_call_payloads(calls: list[dict[str, Any]], repair: bool) -> list[dict[str, Any]]:
    payloads = []
    for call in calls:
        arguments = parse_tool_args(call["arguments"]) if repair else call["arguments"]
        payloads.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    return payloads


def completion_body(
    model_name: str,
    content: str,
    calls: list[dict[str, Any]],
    repair: bool,
    prompt_tokens: int,
    new_tokens: int,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant"}
    if calls:
        message["content"] = content or None
        message["tool_calls"] = tool_call_payloads(calls, repair)
        finish_reason = "tool_calls"
    else:
        message["content"] = content
        finish_reason = "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason, "logprobs": None}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": new_tokens,
            "total_tokens": prompt_tokens + new_tokens,
        },
    }


def stream_chunks(
    model_name: str,
    content: str,
    calls: list[dict[str, Any]],
    repair: bool,
) -> list[dict[str, Any]]:
    """SSE chunks in the shape `llama_cpp/entity.py:_transform_stream` reads.

    It wants: a first chunk carrying `delta.role`, then content deltas, then one
    chunk per tool call whose delta holds `index`, `id` and `function.name` /
    `.arguments`, then a chunk with a `finish_reason` (which is where it flushes the
    call it was accumulating). We generate the whole answer first and then emit it,
    so this is the same answer the non-streaming path gives - the stream is there
    because the integration turns it on when the server supports it, not to save
    time.
    """
    base = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
    }
    chunks: list[dict[str, Any]] = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    ]
    if content:
        chunks.append(
            {**base, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
        )
    for index, payload in enumerate(tool_call_payloads(calls, repair)):
        chunks.append(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": payload["id"],
                                    "type": "function",
                                    "function": payload["function"],
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )
    chunks.append(
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if calls else "stop",
                }
            ],
        }
    )
    return chunks


# ---------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------


class State:
    """Everything the handler needs, and the lock that keeps the GPU to one caller."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = Model(args.model_path, args.device)
        self.lock = threading.Lock()
        self.requests = 0
        self.total_ms = 0.0
        self.started = time.time()


STATE: State | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "hua-serve/1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        """Silence the default stderr line; we print our own, with the numbers."""

    # -- plumbing -----------------------------------------------------------

    def _send_json(self, code: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, code: int, message: str, err_type: str = "invalid_request_error") -> None:
        self._send_json(code, {"error": {"message": message, "type": err_type, "code": code}})

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length))

    # -- routes -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        assert STATE is not None
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path in ("/v1/models", "/models"):
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": STATE.args.model_name,
                            "object": "model",
                            "created": int(STATE.started),
                            "owned_by": "local",
                        }
                    ],
                },
            )
        elif path == "/health":
            mean = STATE.total_ms / STATE.requests if STATE.requests else 0.0
            self._send_json(
                200,
                {
                    "model_name": STATE.args.model_name,
                    "model_path": STATE.args.model_path,
                    "precision": "bfloat16",
                    "device": STATE.args.device,
                    "requests": STATE.requests,
                    "mean_ms": round(mean, 1),
                    "peak_gib": round(STATE.model.peak_gib(), 2),
                    "max_new_tokens": STATE.args.max_new_tokens,
                    "greedy": bool(STATE.args.greedy),
                    "temperature_override": STATE.args.temperature,
                    "repair_args": bool(STATE.args.repair_args),
                    "uptime_s": round(time.time() - STATE.started, 1),
                },
            )
        else:
            self._send_error_json(404, f"no route {path}")

    def do_POST(self) -> None:  # noqa: N802
        assert STATE is not None
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_error_json(404, f"no route {path}")
            return
        try:
            request = self._read_json()
        except ValueError as err:
            self._send_error_json(400, f"body is not json: {err}")
            return
        if not isinstance(request, dict):
            self._send_error_json(400, "body is not a json object")
            return

        args = STATE.args
        wanted = request.get("model")
        if wanted and wanted != args.model_name and not args.any_model:
            # A client pointed at the wrong server fails loudly instead of being
            # answered by other weights.
            self._send_error_json(
                404,
                f"this server holds '{args.model_name}', not '{wanted}'",
                err_type="model_not_found",
            )
            log(f"REFUSED a request for model '{wanted}'; this server holds '{args.model_name}'")
            return

        messages = request.get("messages") or []
        tools = request.get("tools") or None
        stream = bool(request.get("stream"))
        cap = args.max_new_tokens
        asked = request.get("max_tokens") or request.get("max_completion_tokens")
        max_new_tokens = min(int(asked), cap) if asked else cap

        # Sampling: by default exactly what the client sends. Home Assistant's
        # llama_cpp integration sends temperature 0.7 and top_p 1.0 unless its
        # settings override them. --greedy or --temperature override the client.
        temperature = request.get("temperature")
        if args.temperature is not None:
            temperature = args.temperature
        if temperature is None:
            temperature = 1.0
        temperature = float(temperature)
        top_p = float(request.get("top_p") or 1.0)
        do_sample = not args.greedy and temperature > 0.0

        started = time.monotonic()
        try:
            with STATE.lock:
                text = render_chat(STATE.model.tokenizer, messages, tools)
                completion, prompt_tokens, new_tokens, truncated = STATE.model.generate(
                    text, max_new_tokens, do_sample, temperature, top_p, args.top_k
                )
        except Exception as err:  # noqa: BLE001 - a 500 tells the harness, a crash does not
            log(f"generation failed: {err!r}")
            self._send_error_json(500, repr(err), err_type="server_error")
            return
        ms = (time.monotonic() - started) * 1000
        STATE.requests += 1
        STATE.total_ms += ms

        content, calls, notes = split_completion(completion)
        names = ",".join(call["name"] for call in calls) or "-"
        log(
            f"POST /v1/chat/completions  {ms:7.0f} ms  prompt {prompt_tokens:5d} tok  "
            f"new {new_tokens:4d} tok  tools_in {len(tools or [])}  msgs {len(messages)}  "
            f"temp {temperature:.2f} top_p {top_p:.2f} sample {int(do_sample)} stream {int(stream)}  "
            f"calls {names}"
            + ("  TRUNCATED" if truncated else "")
            + (f"  notes: {'; '.join(notes)}" if notes and not calls else "")
        )

        if not stream:
            self._send_json(
                200,
                completion_body(args.model_name, content, calls, args.repair_args, prompt_tokens, new_tokens),
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for chunk in stream_chunks(args.model_name, content, calls, args.repair_args):
            self._write_chunk(f"data: {json.dumps(chunk)}\n\n")
        self._write_chunk("data: [DONE]\n\n")
        self._write_chunk("")  # the terminating zero-length chunk

    def _write_chunk(self, text: str) -> None:
        body = text.encode()
        self.wfile.write(f"{len(body):X}\r\n".encode())
        self.wfile.write(body)
        self.wfile.write(b"\r\n")
        self.wfile.flush()


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"a Hugging Face repo id or a local folder (default {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"the id this server answers to; must match the model picked in Home Assistant (default {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="cap; Home Assistant asks for 3000")
    parser.add_argument("--top-k", type=int, default=0, help="0 = off, as the scores were run")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="override the temperature the client sends (llama_cpp sends 0.7)",
    )
    parser.add_argument("--greedy", action="store_true", help="do_sample=False whatever the client asks for")
    parser.add_argument(
        "--repair-args",
        action="store_true",
        help="apply Home Assistant's ollama-only argument repair (drop empty args, "
        "parse stringified json). Off by default: the llama_cpp integration does not do it.",
    )
    parser.add_argument("--any-model", action="store_true", help="do not refuse a mismatched model field")
    args = parser.parse_args(argv)

    global STATE  # noqa: PLW0603 - one model, one process
    STATE = State(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    log(
        f"serving '{args.model_name}' on http://{args.host}:{args.port}/v1  "
        f"(weights {args.model_path}, bf16, "
        f"max_new_tokens {args.max_new_tokens}, greedy {bool(args.greedy)}, "
        f"repair_args {bool(args.repair_args)})"
    )
    log(f"base_url for Home Assistant: http://<this host>:{args.port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("stopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
