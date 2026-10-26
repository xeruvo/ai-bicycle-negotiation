"""Offline tests for the proposed judge helpers with AI assistance."""

from copy import deepcopy
from unittest.mock import patch

import requests

from judge import build_judge_messages, judge, parse_judgement
from llm_client import ChatResponse


RUBRIC = "Judge the negotiation using the supplied evidence."
JUDGE_CONFIG = {
    "model": "llama3.2:3b",
    "temperature": 0,
    "budget": {"max_turns": 2, "max_tokens": 100, "max_seconds": 10},
    "max_input_bytes": 7000,
    "system_prompt": RUBRIC,
}


def source_record():
    return {
        "meta": {
            "config": {
                "topic": "Used bicycle price",
                "agents": [
                    {"name": "Buyer", "system_prompt": "Buyer persona"},
                    {"name": "Seller", "system_prompt": "Seller persona"},
                ],
            }
        },
        "messages": [
            {
                "speaker": "Buyer",
                "content": "I offer 700 kr.",
                "parse_valid": True,
            }
        ],
    }


def response(text, prompt_tokens=2, completion_tokens=3, seconds=0.1):
    return ChatResponse(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        seconds=seconds,
    )


class SequenceClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def chat(self, model, messages, temperature=0.7, timeout=300):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "timeout": timeout,
            }
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class ControlledClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value


def test_judge_valid_first_reply_and_accounting():
    record = source_record()
    original = deepcopy(record)
    client = SequenceClient(
        [response('{"reason": "Turn 1 offered 700.", "success": true, "score": 5}')]
    )

    evaluation = judge(record, client, JUDGE_CONFIG)

    assert evaluation["status"] == "complete"
    assert evaluation["verdict"]["score"] == 5
    assert len(evaluation["attempts"]) == 1
    assert evaluation["attempts"][0]["input_tokens"] == 2
    assert evaluation["attempts"][0]["output_tokens"] == 3
    assert evaluation["totals"] == {
        "attempts": 1,
        "input_tokens": 2,
        "output_tokens": 3,
        "seconds": 0.1,
    }
    assert client.calls[0]["model"] == "llama3.2:3b"
    assert client.calls[0]["temperature"] == 0
    assert 0 < client.calls[0]["timeout"] <= 10
    assert record == original


def test_judge_recovers_once_from_malformed_reply():
    malformed_output = "not JSON\nwith exact spacing"
    client = SequenceClient(
        [
            response(malformed_output, prompt_tokens=2, completion_tokens=3),
            response('{"reason": "Turn 2 accepted.", "success": true, "score": 4}'),
        ]
    )

    evaluation = judge(source_record(), client, JUDGE_CONFIG)

    assert evaluation["status"] == "complete"
    assert evaluation["verdict"]["score"] == 4
    assert len(evaluation["attempts"]) == 2
    assert evaluation["attempts"][0]["parsing_error"]
    assert evaluation["attempts"][1]["parsing_error"] is None
    retry_messages = client.calls[1]["messages"][-2:]
    assert retry_messages[0] == {"role": "assistant", "content": malformed_output}
    assert retry_messages[1]["role"] == "user"
    assert "malformed" in retry_messages[1]["content"]
    assert evaluation["totals"] == {
        "attempts": 2,
        "input_tokens": 4,
        "output_tokens": 6,
        "seconds": 0.2,
    }


def test_judge_does_not_retry_after_token_budget_exhaustion():
    client = SequenceClient([response("not JSON", prompt_tokens=2, completion_tokens=3)])
    config = deepcopy(JUDGE_CONFIG)
    config["budget"]["max_tokens"] = 5

    evaluation = judge(source_record(), client, config)

    assert evaluation["status"] == "max_tokens"
    assert evaluation["verdict"] is None
    assert len(evaluation["attempts"]) == 1
    assert evaluation["totals"]["input_tokens"] == 2
    assert evaluation["totals"]["output_tokens"] == 3
    assert len(client.calls) == 1


def test_judge_does_not_retry_when_retry_history_exceeds_input_limit():
    malformed_output = "not JSON\nwith exact spacing"
    client = SequenceClient([response(malformed_output)])
    config = deepcopy(JUDGE_CONFIG)
    base_messages = build_judge_messages(source_record(), RUBRIC)
    base_bytes = sum(
        len(message["content"].encode("utf-8")) for message in base_messages
    )
    correction_bytes = len(
        (
            "Your previous output was malformed: "
            "Invalid judge JSON: Expecting value: line 1 column 1 (char 0) "
            "Return exactly one valid JSON object with reason, success, and score."
        ).encode("utf-8")
    )
    config["max_input_bytes"] = base_bytes + len(malformed_output.encode("utf-8")) + correction_bytes - 1

    evaluation = judge(source_record(), client, config)

    assert evaluation["status"] == "input_limit"
    assert evaluation["verdict"] is None
    assert len(evaluation["attempts"]) == 1
    assert len(client.calls) == 1


def test_judge_stops_after_two_malformed_replies():
    client = SequenceClient([response("bad one"), response("bad two")])

    evaluation = judge(source_record(), client, JUDGE_CONFIG)

    assert evaluation["status"] == "malformed"
    assert evaluation["verdict"] is None
    assert len(evaluation["attempts"]) == 2
    assert evaluation["budget"]["stop_reason"] == "max_turns"


def test_judge_records_timeout_without_fabricating_verdict():
    client = SequenceClient([requests.Timeout("judge request timed out")])

    evaluation = judge(source_record(), client, JUDGE_CONFIG)

    assert evaluation["status"] == "request_timeout"
    assert evaluation["verdict"] is None
    assert evaluation["attempts"] == []
    assert evaluation["request_errors"][0]["exception_type"] == "Timeout"
    assert "unknown" in evaluation["request_errors"][0]["explanation"]


def test_judge_rejects_late_response_after_time_budget():
    clock = ControlledClock()
    client = SequenceClient(
        [response('{"reason": "late", "success": true, "score": 5}')]
    )
    original_config = deepcopy(JUDGE_CONFIG)
    config = deepcopy(JUDGE_CONFIG)
    config["budget"]["max_seconds"] = 1

    def advance_before_response(model, messages, temperature=0.7, timeout=300):
        clock.value = 2
        return client.outcomes.pop(0)

    client.chat = advance_before_response
    with patch("budget.time.monotonic", side_effect=clock.monotonic):
        evaluation = judge(source_record(), client, config)

    assert evaluation["status"] == "max_seconds"
    assert evaluation["verdict"] is None
    assert len(evaluation["attempts"]) == 1
    assert evaluation["attempts"][0]["raw_text"]
    assert config == {**original_config, "budget": {**original_config["budget"], "max_seconds": 1}}


def test_judge_rejects_oversized_input_without_calling_client():
    client = SequenceClient([])
    config = deepcopy(JUDGE_CONFIG)
    config["max_input_bytes"] = 10

    evaluation = judge(source_record(), client, config)

    assert evaluation["status"] == "input_limit"
    assert evaluation["verdict"] is None
    assert evaluation["attempts"] == []
    assert client.calls == []


def test_valid_judgements():
    assert parse_judgement(
        '{"reason": "Turn 2 accepted the latest offer.", "success": true, "score": 5}'
    ) == {
        "reason": "Turn 2 accepted the latest offer.",
        "success": True,
        "score": 5,
    }
    assert parse_judgement(
        '{"reason": "No valid agreement was reached.", "success": false, "score": 3}'
    )["success"] is False


def assert_rejected(text):
    try:
        parse_judgement(text)
    except ValueError:
        return
    raise AssertionError(f"judge output should be rejected: {text}")


def test_malformed_missing_and_extra_fields_are_rejected():
    assert_rejected("not JSON")
    assert_rejected('{"reason": "missing fields"}')
    assert_rejected(
        '{"reason": "extra", "success": false, "score": 2, "extra": true}'
    )


def test_every_score_requires_matching_success():
    for score in range(1, 6):
        expected_success = score >= 4
        valid = (
            f'{{"reason": "score {score}", "success": '
            f'{str(expected_success).lower()}, "score": {score}}}'
        )
        assert parse_judgement(valid)["score"] == score

        opposite = not expected_success
        invalid = (
            f'{{"reason": "score {score}", "success": '
            f'{str(opposite).lower()}, "score": {score}}}'
        )
        assert_rejected(invalid)
def test_wrong_types_and_out_of_range_scores_are_rejected():
    assert_rejected('{"reason": 4, "success": true, "score": 4}')
    assert_rejected('{"reason": "wrong", "success": "true", "score": 4}')
    assert_rejected('{"reason": "wrong", "success": true, "score": true}')
    assert_rejected('{"reason": "low", "success": false, "score": 0}')
    assert_rejected('{"reason": "high", "success": true, "score": 6}')
    assert_rejected('{"reason": "", "success": false, "score": 2}')
    assert_rejected('{"reason": "   ", "success": false, "score": 2}')


def test_inconsistent_score_and_success_are_rejected():
    assert_rejected('{"reason": "inconsistent", "success": true, "score": 3}')
    assert_rejected('{"reason": "inconsistent", "success": false, "score": 4}')


def test_judge_messages_preserve_evidence_and_exclude_previous_verdicts():
    record = {
        "meta": {
            "config": {
                "topic": "Used bicycle price",
                "agents": [
                    {"name": "Buyer", "system_prompt": "Buyer persona"},
                    {"name": "Seller", "system_prompt": "Seller persona"},
                ],
            },
            "judge_score": 1,
            "negotiation_outcome": "old outcome",
        },
        "judge": {"reason": "old verdict", "success": True, "score": 5},
        "negotiation_outcome": {"success": True},
        "messages": [
            {
                "speaker": "Buyer",
                "content": "I offer 700 kr.",
                "parse_valid": True,
            },
            {
                "speaker": "Seller",
                "content": "I accept 700 kr.",
                "parse_valid": False,
            },
        ],
    }
    original = deepcopy(record)

    messages = build_judge_messages(record, RUBRIC)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert RUBRIC in messages[0]["content"]
    evidence = messages[1]["content"]
    assert "Used bicycle price" in evidence
    assert "Buyer persona" in evidence
    assert "Seller persona" in evidence
    assert "Turn 1" in evidence
    assert "Turn 2" in evidence
    assert evidence.index("I offer 700 kr.") < evidence.index("I accept 700 kr.")
    assert "parse_valid: True" in evidence
    assert "parse_valid: False" in evidence
    assert "old verdict" not in evidence
    assert "old outcome" not in evidence
    assert '"success": True' not in evidence
    evaluation_request = (
        "Evaluate the conversation above against the system rubric. In your reason, "
        "identify the decisive turn numbers and actual prices or actions. Explain "
        "whether the final agreement is valid, and identify any opening-offer or "
        "price-limit violation. Check your claims against the quoted conversation: "
        "do not say a price was omitted if the speaker explicitly stated it. Do not "
        "copy a score definition as your reason. Return exactly one JSON object with "
        "reason, success, and score."
    )
    assert evaluation_request in evidence
    assert evidence.index("END NEGOTIATION EVIDENCE") < evidence.index(
        evaluation_request
    )
    assert record == original


if __name__ == "__main__":
    test_valid_judgements()
    test_every_score_requires_matching_success()
    test_malformed_missing_and_extra_fields_are_rejected()
    test_wrong_types_and_out_of_range_scores_are_rejected()
    test_inconsistent_score_and_success_are_rejected()
    test_judge_messages_preserve_evidence_and_exclude_previous_verdicts()
    test_judge_valid_first_reply_and_accounting()
    test_judge_recovers_once_from_malformed_reply()
    test_judge_does_not_retry_after_token_budget_exhaustion()
    test_judge_does_not_retry_when_retry_history_exceeds_input_limit()
    test_judge_stops_after_two_malformed_replies()
    test_judge_records_timeout_without_fabricating_verdict()
    test_judge_rejects_late_response_after_time_budget()
    test_judge_rejects_oversized_input_without_calling_client()
    print("Judge helper and call checks passed.")
