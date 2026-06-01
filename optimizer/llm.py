"""Minimal LLM client — the one swappable model boundary.

Two backends, selected by the LLM_BACKEND env var:

  LLM_BACKEND=ollama  (default)  -> Ollama /api/generate at OLLAMA_HOST
  LLM_BACKEND=openai             -> OpenAI-compatible /v1/chat/completions at
                                    OPENAI_BASE (e.g. a llama.cpp `llama-server`,
                                    LM Studio, or any OpenAI-style endpoint)

The openai backend lets the optimizer reuse a model server you already run
(e.g. Localpen's llama-server on :11434) instead of loading a second copy of a
7B model into RAM.

Not unit-tested against the network; the optimizer loop is tested with a stubbed
generate(). Configure via env vars:
  LLM_BACKEND, MODEL, OLLAMA_HOST, OPENAI_BASE
"""

import json
import os
import platform
import time
import urllib.error
import urllib.request

BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()
MODEL = os.environ.get("MODEL", "qwen2.5-coder:7b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# Base URL for the OpenAI-compatible backend (note the /v1 suffix).
OPENAI_BASE = os.environ.get("OPENAI_BASE", "http://localhost:11434/v1")


class OllamaError(RuntimeError):
    """Raised when the model server is unreachable or misconfigured.

    (Name kept for backwards compatibility; covers both backends.)"""


LLMError = OllamaError  # clearer alias for the openai backend


def _post_json(url: str, payload: dict, timeout: int):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _install_hint() -> str:
    if platform.system() == "Darwin":
        return "Install: `brew install ollama` then `ollama serve`."
    if platform.system() == "Linux":
        return "Install from https://ollama.com/download then `ollama serve`."
    return "See https://ollama.com/download to install, then start the server."


# --- Ollama backend ---------------------------------------------------------

def _preflight_ollama() -> None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as r:
            tags = json.loads(r.read())
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_HOST} ({e}). {_install_hint()}"
        )
    names = {m.get("name", "") for m in tags.get("models", [])}
    if MODEL not in names and f"{MODEL}:latest" not in names:
        raise OllamaError(
            f"Model {MODEL!r} not found on the Ollama server. "
            f"Pull it with: `ollama pull {MODEL}`."
        )


def _generate_ollama(prompt: str, temperature: float) -> str:
    data = _post_json(f"{OLLAMA_HOST}/api/generate", {
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature},
    }, timeout=180)
    return data["response"]


# --- OpenAI-compatible backend (llama.cpp, LM Studio, ...) ------------------

def _preflight_openai() -> None:
    try:
        with urllib.request.urlopen(f"{OPENAI_BASE}/models", timeout=5) as r:
            json.loads(r.read())
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Cannot reach an OpenAI-compatible server at {OPENAI_BASE} ({e}). "
            f"Start your model server (e.g. Localpen / llama-server) or set "
            f"OPENAI_BASE to its /v1 URL."
        )


def _generate_openai(prompt: str, temperature: float) -> str:
    data = _post_json(f"{OPENAI_BASE}/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "stream": False,
    }, timeout=180)
    return data["choices"][0]["message"]["content"]


# --- Public API -------------------------------------------------------------

def preflight() -> None:
    """Raise OllamaError with a helpful message if the server isn't ready."""
    if BACKEND == "openai":
        _preflight_openai()
    else:
        _preflight_ollama()


def generate(prompt: str, temperature: float = 0.8, retries: int = 3) -> str:
    """Generate text from the configured backend. Retries with backoff."""
    backend = _generate_openai if BACKEND == "openai" else _generate_ollama
    last = None
    for attempt in range(retries):
        try:
            return backend(prompt, temperature)
        except (urllib.error.URLError, OSError, KeyError, IndexError,
                TimeoutError) as e:
            # OSError covers socket.timeout on Python <= 3.10 (a mid-transfer
            # stall), which is not wrapped in URLError; retry rather than escape.
            last = e
            time.sleep(2 * (attempt + 1))
    raise OllamaError(f"generate failed after {retries} tries: {last}")
