"""Tests for Ollama request timeout handling without network access."""

import math
from unittest.mock import Mock, patch

import requests

from llm_client import OllamaClient


MESSAGES = [{"role": "user", "content": "Hello"}]


def response_mock():
    response = Mock()
    response.json.return_value = {
        "message": {"content": "reply"},
        "prompt_eval_count": 4,
        "eval_count": 2,
    }
    return response


def test_supplied_timeout_reaches_request():
    client = OllamaClient()

    with patch("llm_client.requests.post", return_value=response_mock()) as post:
        client.chat("test-model", MESSAGES, timeout=2.5)

    assert post.call_args.kwargs["timeout"] == 2.5


def test_omitted_timeout_preserves_default():
    client = OllamaClient()

    with patch("llm_client.requests.post", return_value=response_mock()) as post:
        client.chat("test-model", MESSAGES)

    assert post.call_args.kwargs["timeout"] == 300


def test_invalid_timeout_is_rejected_before_request():
    invalid_values = [None, 0, -1, math.inf, math.nan, True, "300"]

    for timeout in invalid_values:
        client = OllamaClient()
        with patch("llm_client.requests.post") as post:
            try:
                client.chat("test-model", MESSAGES, timeout=timeout)
            except ValueError:
                pass
            else:
                raise AssertionError(f"timeout should be rejected: {timeout!r}")
            post.assert_not_called()


def test_requests_timeout_reaches_caller():
    client = OllamaClient()
    timeout_error = requests.Timeout("request timed out")

    with patch("llm_client.requests.post", side_effect=timeout_error):
        try:
            client.chat("test-model", MESSAGES, timeout=2.5)
        except requests.Timeout as error:
            assert error is timeout_error
        else:
            raise AssertionError("requests.Timeout should reach the caller")


if __name__ == "__main__":
    test_supplied_timeout_reaches_request()
    test_omitted_timeout_preserves_default()
    test_invalid_timeout_is_rejected_before_request()
    test_requests_timeout_reaches_caller()
    print("Client timeout checks passed.")