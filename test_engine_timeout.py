"""Deterministic engine request-timeout checks with AI assistance."""

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import requests

from agents import Agent
from budget import Budget
from engine import DialogueEngine
from llm_client import ChatResponse


AGENTS = [
    Agent("Buyer", "Buyer timeout test prompt."),
    Agent("Seller", "Seller timeout test prompt."),
]


class ControlledClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class DeterministicClient:
    def __init__(self, clock, outcomes):
        self.clock = clock
        self.outcomes = list(outcomes)
        self.timeouts = []

    def chat(self, model, messages, temperature=0.7, timeout=300):
        self.timeouts.append(timeout)
        outcome, duration = self.outcomes.pop(0)
        self.clock.advance(duration)
        if isinstance(outcome, BaseException):
            raise outcome
        outcome.seconds = duration
        return outcome


def response(text="reply", prompt_tokens=2, completion_tokens=3):
    return ChatResponse(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        seconds=0,
    )


def run_with_clock(engine, clock):
    with patch("budget.time.monotonic", side_effect=clock.monotonic):
        return engine.run()


def test_timeout_decreases_as_time_is_consumed():
    clock = ControlledClock()
    client = DeterministicClient(
        clock,
        [(response(f"reply {number}"), 1) for number in range(3)],
    )
    engine = DialogueEngine(
        AGENTS,
        client,
        Budget(max_turns=3, max_tokens=100, max_seconds=10),
    )

    run_with_clock(engine, clock)

    assert client.timeouts == [10, 9, 8]
    assert engine.budget.stop_reason == "max_turns"


def test_context_preparation_can_consume_remaining_time():
    clock = ControlledClock()
    client = DeterministicClient(clock, [])

    def consume_remaining_time(messages):
        clock.advance(5)
        return messages

    engine = DialogueEngine(
        AGENTS,
        client,
        Budget(max_turns=3, max_tokens=100, max_seconds=5),
        manage_context=consume_remaining_time,
    )

    run_with_clock(engine, clock)

    assert client.timeouts == []
    assert engine.transcript == []
    assert engine.budget.stop_reason == "max_seconds"


def test_completed_reply_then_timeout_preserves_completed_accounting():
    clock = ControlledClock()
    timeout_error = requests.Timeout("request timed out")
    client = DeterministicClient(
        clock,
        [(response(), 1), (timeout_error, 2)],
    )
    engine = DialogueEngine(
        AGENTS,
        client,
        Budget(max_turns=5, max_tokens=100, max_seconds=10),
    )

    run_with_clock(engine, clock)

    assert len(engine.transcript) == 1
    assert engine.transcript[0].content == "reply"
    assert engine.budget.turns == 1
    assert engine.budget.tokens == 5
    assert engine.budget.stop_reason == "request_timeout"
    assert client.timeouts == [10, 9]
    assert engine.request_errors[0] == {
        "speaker": "Seller",
        "turn_index": 1,
        "timeout": 9,
        "request_duration": 2,
        "exception_type": "Timeout",
        "explanation": "Token usage for the unfinished request is unknown.",
    }


def test_timeout_stop_reason_depends_on_elapsed_budget():
    for duration, expected_reason in [(1, "request_timeout"), (10, "max_seconds")]:
        clock = ControlledClock()
        client = DeterministicClient(
            clock,
            [(requests.Timeout("request timed out"), duration)],
        )
        engine = DialogueEngine(
            AGENTS,
            client,
            Budget(max_turns=5, max_tokens=100, max_seconds=10),
        )

        run_with_clock(engine, clock)

        assert engine.transcript == []
        assert engine.budget.stop_reason == expected_reason
        assert len(engine.request_errors) == 1


def test_late_completed_reply_is_counted_before_max_seconds_stop():
    clock = ControlledClock()
    client = DeterministicClient(clock, [(response(), 6)])
    agreement_checks = []

    def goal_reached(transcript):
        agreement_checks.append(transcript)
        return True

    engine = DialogueEngine(
        AGENTS,
        client,
        Budget(max_turns=5, max_tokens=100, max_seconds=5),
        goal_reached=goal_reached,
    )

    run_with_clock(engine, clock)

    assert len(engine.transcript) == 1
    assert engine.budget.turns == 1
    assert engine.budget.tokens == 5
    assert engine.budget.stop_reason == "max_seconds"
    assert agreement_checks == []


def test_timeout_evidence_is_saved_and_reopened():
    clock = ControlledClock()
    client = DeterministicClient(
        clock,
        [(response(), 1), (requests.Timeout("request timed out"), 2)],
    )
    engine = DialogueEngine(
        AGENTS,
        client,
        Budget(max_turns=5, max_tokens=100, max_seconds=10),
    )

    evidence_path = Path("transcripts") / f"engine_timeout_mock_{uuid4().hex}.json"
    with patch("budget.time.monotonic", side_effect=clock.monotonic):
        engine.run()
        engine.save(
            evidence_path,
            meta={
                "mode": "mock development test",
                "test_name": "engine_timeout",
                "clock": "simulated",
            },
        )
        with evidence_path.open() as file:
            saved = json.load(file)

    assert saved["request_errors"] == engine.request_errors
    assert saved["messages"] == [asdict(entry) for entry in engine.transcript]
    assert saved["budget"]["stop_reason"] == "request_timeout"
    assert saved["budget"]["seconds"] == 3
    assert saved["totals"]["seconds"] == 1
    assert saved["request_errors"][0]["request_duration"] == 2
    assert saved["budget"]["turns"] == 1
    assert saved["budget"]["tokens"] == 5
    assert saved["totals"]["turns"] == 1
    return evidence_path


if __name__ == "__main__":
    test_timeout_decreases_as_time_is_consumed()
    test_context_preparation_can_consume_remaining_time()
    test_completed_reply_then_timeout_preserves_completed_accounting()
    test_timeout_stop_reason_depends_on_elapsed_budget()
    test_late_completed_reply_is_counted_before_max_seconds_stop()
    evidence_path = test_timeout_evidence_is_saved_and_reopened()
    print(f"Engine timeout checks passed. Evidence path: {evidence_path}")
