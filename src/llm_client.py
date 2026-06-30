"""Shared LLM client — OpenAI-compatible, wall-clock timeout, global concurrency cap."""
import concurrent.futures
import json
import os
import re
import threading

import httpx
from openai import OpenAI


# ── Config ───────────────────────────────────────────────────────────────────
BASE_URL     = os.getenv("LLM_BASE_URL",    "http://localhost:11434/v1")
API_KEY      = os.getenv("LLM_API_KEY",     "ollama")
MODEL        = os.getenv("LLM_MODEL",       "gemma2:27b")
TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE",  "0.0"))
MAX_TOKENS   = int(os.getenv("LLM_OUTPUT_TOKENS",  "2048"))
WALL_TIMEOUT = int(os.getenv("LLM_WALL_TIMEOUT",   "300"))

_GLOBAL_CONCURRENCY = int(os.getenv("LLM_GLOBAL_CONCURRENCY", "4"))
_global_sem         = threading.Semaphore(_GLOBAL_CONCURRENCY)


def make_client() -> OpenAI:
    # Fresh client per call — keepalive disabled to prevent silent stale-connection hangs.
    return OpenAI(
        api_key     = API_KEY,
        base_url    = BASE_URL,
        http_client = httpx.Client(
            limits  = httpx.Limits(
                max_keepalive_connections = 0,
                max_connections           = 1,
            ),
            timeout = httpx.Timeout(600.0, connect=10.0),
        ),
    )


def call_llm(
    messages:     list,
    model:        str  = "",
    temperature:  float = None,
    max_tokens:   int   = None,
    wall_timeout: int   = None,
    json_mode:    bool  = True,
) -> str:
    _model        = model        or MODEL
    _temperature  = temperature  if temperature  is not None else TEMPERATURE
    _max_tokens   = max_tokens   if max_tokens   is not None else MAX_TOKENS
    _wall_timeout = wall_timeout if wall_timeout is not None else WALL_TIMEOUT

    kwargs: dict = dict(
        model       = _model,
        messages    = messages,
        temperature = _temperature,
        max_tokens  = _max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    def _call(kw=kwargs):
        kw = {**kw, "messages": messages}
        return make_client().chat.completions.create(**kw)

    # Single throttle point for all users/layers.
    _global_sem.acquire()
    pool   = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_call)
    try:
        response = future.result(timeout=_wall_timeout)
        return response.choices[0].message.content or ""
    except concurrent.futures.TimeoutError:
        raise TimeoutError(
            f"LLM did not respond within {_wall_timeout}s — server may be overloaded"
        )
    finally:
        pool.shutdown(wait=False)
        _global_sem.release()


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$",        "", raw)
    return json.loads(raw.strip())


def check_server() -> None:
    server_root = BASE_URL.rstrip("/").removesuffix("v1").rstrip("/")
    try:
        httpx.get(f"{server_root}/health", timeout=8.0)
        return
    except Exception:
        pass
    try:
        httpx.get(f"{server_root}/v1/models",
                  headers={"Authorization": f"Bearer {API_KEY}"}, timeout=8.0)
    except Exception as e:
        raise ConnectionError(f"LLM server unreachable at {BASE_URL} — {e}")
