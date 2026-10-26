"""Tests for optional Ollama model settings without network access."""

from unittest.mock import Mock, patch

from llm_client import OllamaClient, make_client


MESSAGES = [{"role": "user", "content": "Hello"}]


def response_mock():
    response = Mock()
    response.json.return_value = {
        "message": {"content": "reply"},
        "prompt_eval_count": 4,
        "eval_count": 2,
    }
    return response


def test_ollama_options_reach_request_and_temperature_wins():
    options = {"num_ctx": 4096, "num_predict": 128, "temperature": 0.1}
    original_options = options.copy()
    client = OllamaClient(options=options)

    with patch("llm_client.requests.post", return_value=response_mock()) as post:
        client.chat("test-model", MESSAGES, temperature=0.8)

    payload = post.call_args.kwargs["json"]
    assert payload["options"] == {
        "num_ctx": 4096,
        "num_predict": 128,
        "temperature": 0.8,
    }
    assert options == original_options
    assert client.options == original_options


def test_make_client_passes_options_to_ollama_client():
    options = {"num_ctx": 4096, "num_predict": 128}

    client = make_client(options=options)

    assert isinstance(client, OllamaClient)
    assert client.options == options
    assert client.options is not options


def test_no_options_preserves_existing_payload_behavior():
    client = OllamaClient()

    with patch("llm_client.requests.post", return_value=response_mock()) as post:
        client.chat("test-model", MESSAGES, temperature=0.6)

    post.assert_called_once_with(
        "http://localhost:11434/api/chat",
        json={
            "model": "test-model",
            "messages": MESSAGES,
            "stream": False,
            "options": {"temperature": 0.6},
        },
        timeout=300,
    )


def test_make_client_response_format_reaches_top_level_request_field():
    client = make_client(response_format="json")

    with patch("llm_client.requests.post", return_value=response_mock()) as post:
        client.chat("test-model", MESSAGES)

    payload = post.call_args.kwargs["json"]
    assert payload["format"] == "json"
    assert "format" not in payload["options"]


def test_mock_client_path_is_unchanged():
    client = make_client(mock=True, options={"num_ctx": 4096})

    assert not isinstance(client, OllamaClient)
    assert client.chat("ignored", MESSAGES).text


if __name__ == "__main__":
    test_ollama_options_reach_request_and_temperature_wins()
    test_make_client_passes_options_to_ollama_client()
    test_no_options_preserves_existing_payload_behavior()
    test_make_client_response_format_reaches_top_level_request_field()
    test_mock_client_path_is_unchanged()
    print("Client option checks passed.")
