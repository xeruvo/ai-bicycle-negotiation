"""llm_client.py: a tiny wrapper over a chat model.

COMPLETE. Two clients with the same interface:
  - OllamaClient: talks to a local model via Ollama's /api/chat.
  - MockClient:   returns canned text instantly and for free (for development).

Both expose:  chat(model, messages, temperature) -> ChatResponse
where `messages` is a list of {"role": "system"|"user"|"assistant", "content": str}.
"""
import math
import queue
import threading
import time
from dataclasses import dataclass

import requests


@dataclass
class ChatResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float

    @property
    def tokens(self):
        return self.prompt_tokens + self.completion_tokens


class OllamaClient:
    """Calls a local model served by Ollama (https://ollama.com).

    Optional model settings and network timeout support were added with AI
    assistance. The timeout is a network request timeout, not a guaranteed
    total wall-clock deadline. Caller waiting is bounded subject to normal
    scheduling overhead; background HTTP/server work may continue, with no
    guarantee of cancellation or exact real-time precision.
    """

    def __init__(self, host="http://localhost:11434", options=None, response_format=None):
        self.host = host
        self.options = dict(options or {})
        self.response_format = response_format

    def chat(self, model, messages, temperature=0.7, timeout=300):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a finite positive number")
        if isinstance(timeout, float) and not math.isfinite(timeout):
            raise ValueError("timeout must be a finite positive number")
        if timeout <= 0:
            raise ValueError("timeout must be a finite positive number")

        t0 = time.monotonic()
        deadline = t0 + timeout
        request_options = dict(self.options)
        request_options["temperature"] = temperature
        request_payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": request_options,
        }
        if self.response_format is not None:
            request_payload["format"] = self.response_format
        result_queue = queue.Queue(maxsize=1)

        def request_worker():
            response = None
            outcome = None
            try:
                response = requests.post(
                    f"{self.host}/api/chat",
                    json=request_payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()
                outcome = (
                    True,
                    ChatResponse(
                        text=data["message"]["content"].strip(),
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        completion_tokens=data.get("eval_count", 0),
                        seconds=time.monotonic() - t0,
                    ),
                )
            except BaseException as exc:
                outcome = (False, exc)
            try:
                if response is not None:
                    response.close()
            except BaseException as cleanup_error:
                if outcome is None or outcome[0]:
                    outcome = (False, cleanup_error)
            result_queue.put(outcome)

        worker = threading.Thread(target=request_worker, daemon=True)
        worker.start()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise requests.Timeout(
                f"Ollama request exceeded caller deadline of {timeout} seconds"
            )
        try:
            succeeded, result = result_queue.get(timeout=remaining)
        except queue.Empty as exc:
            raise requests.Timeout(
                f"Ollama request exceeded caller deadline of {timeout} seconds"
            ) from exc
        if succeeded:
            return result
        raise result


class MockClient:
    """Free, instant, offline. Cycles through canned replies.

    Token counts are approximated by word count, good enough to exercise your
    Budget logic while you debug the loop. Swap to OllamaClient for real runs.
    """

    DEFAULT_REPLIES = [
        "I think we should weigh the costs before anything else.",
        "Fair, but the long-term benefits clearly outweigh those costs.",
        "Only if we ignore the people who can't adapt quickly.",
        "We can support them with a transition plan; that's solvable.",
        "Then let's agree the plan matters as much as the decision.",
        "Agreed. I think that's our common ground.",
    ]

    def __init__(self, replies=None):
        self._replies = replies or self.DEFAULT_REPLIES
        self._i = 0

    def chat(self, model, messages, temperature=0.7, timeout=300):
        text = self._replies[self._i % len(self._replies)]
        self._i += 1
        prompt_tokens = sum(len(m["content"].split()) for m in messages)
        return ChatResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=max(1, len(text.split())),
            seconds=0.01,
        )


def make_client(
    mock=False, host="http://localhost:11434", options=None, response_format=None
):
    return (
        MockClient()
        if mock
        else OllamaClient(
            host=host, options=options, response_format=response_format
        )
    )
