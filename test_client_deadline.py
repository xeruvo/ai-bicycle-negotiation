"""Deterministic caller-deadline checks with mocked HTTP and AI assistance."""

import time
import threading
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


def test_blocked_request_has_bounded_caller_wait():
    request_started = threading.Event()
    request_returned = threading.Event()
    release_request = threading.Event()
    response_closed = threading.Event()
    response = response_mock()
    response.close.side_effect = response_closed.set

    def blocked_post(*args, **kwargs):
        request_started.set()
        release_request.wait(1)
        request_returned.set()
        return response

    client = OllamaClient()
    caller_finished = threading.Event()
    caller_result = {}

    def invoke_client():
        try:
            client.chat("test-model", MESSAGES, timeout=0.15)
        except BaseException as error:
            caller_result["error"] = error
        finally:
            caller_finished.set()

    started_at = time.monotonic()
    with patch("llm_client.requests.post", side_effect=blocked_post):
        caller = threading.Thread(target=invoke_client)
        try:
            caller.start()
            assert request_started.wait(1)
            assert caller_finished.wait(1)
            elapsed = time.monotonic() - started_at

            assert elapsed >= 0.1
            assert elapsed < 1
            assert isinstance(caller_result["error"], requests.Timeout)
            assert "caller deadline" in str(caller_result["error"])
            assert not request_returned.is_set()
            assert not response_closed.is_set()
        finally:
            release_request.set()
            assert response_closed.wait(1)
            caller.join(1)
            assert not caller.is_alive()
    assert response.close.called
    return elapsed


def test_timely_response_reaches_caller_and_closes_response():
    response = response_mock()
    client = OllamaClient()

    with patch("llm_client.requests.post", return_value=response) as post:
        result = client.chat("test-model", MESSAGES, timeout=1)

    assert result.text == "reply"
    assert result.prompt_tokens == 4
    assert result.completion_tokens == 2
    assert post.call_args.kwargs["timeout"] == 1
    response.close.assert_called_once_with()


def test_request_exception_reaches_caller():
    client = OllamaClient()
    original_error = requests.ConnectionError("connection failed")

    with patch("llm_client.requests.post", side_effect=original_error):
        try:
            client.chat("test-model", MESSAGES, timeout=1)
        except requests.ConnectionError as error:
            assert error is original_error
        else:
            raise AssertionError("the original request exception should propagate")


if __name__ == "__main__":
    measured_wait = test_blocked_request_has_bounded_caller_wait()
    test_timely_response_reaches_caller_and_closes_response()
    test_request_exception_reaches_caller()
    print(f"Measured caller wait: {measured_wait:.3f} seconds")
    print("Client deadline checks passed.")
