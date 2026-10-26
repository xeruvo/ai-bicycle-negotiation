"""Tests for bounded context handling, including a 30-turn engine stress run."""

import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from agents import Agent
from budget import Budget
from context_manager import ContextLimitError, SlidingWindow
from engine import DialogueEngine
from llm_client import ChatResponse


class RecordingMockClient:
    """Return short replies while retaining every request for inspection."""

    def __init__(self):
        self.requests = []
        self.reply_number = 0

    def chat(self, model, messages, temperature=0.7, timeout=300):
        self.requests.append(deepcopy(messages))
        self.reply_number += 1
        return ChatResponse(
            text=f"short reply {self.reply_number}",
            prompt_tokens=sum(len(message["content"].split()) for message in messages),
            completion_tokens=3,
            seconds=0.01,
        )


def test_short_input_is_unchanged_without_mutation():
    manager = SlidingWindow()
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Hello"},
    ]
    original = deepcopy(messages)

    retained = manager(messages)

    assert retained == messages
    assert retained is not messages
    assert all(left is not right for left, right in zip(retained, messages))
    assert messages == original
    assert manager.events == []


def test_count_limit_keeps_recent_messages_in_order_and_roles():
    manager = SlidingWindow(max_messages=3, max_content_bytes=3000)
    messages = [{"role": "system", "content": "System"}]
    messages.extend(
        {"role": role, "content": f"message {number}"}
        for number, role in enumerate(
            ["user", "assistant", "user", "assistant", "user"], start=1
        )
    )
    original = deepcopy(messages)

    retained = manager(messages)

    assert retained == [messages[0], *messages[-3:]]
    assert [message["role"] for message in retained] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages == original
    assert manager.events[0]["original_count"] == 6
    assert manager.events[0]["retained_count"] == 4
    assert manager.events[0]["dropped_count"] == 2
    assert manager.events[0]["strategy"] == "truncation"


def test_byte_limit_removes_old_messages_before_count_limit():
    manager = SlidingWindow(max_messages=10, max_content_bytes=8)
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "older"},
        {"role": "assistant", "content": "aa"},
        {"role": "user", "content": "bb"},
        {"role": "assistant", "content": "cc"},
    ]

    retained = manager(messages)

    assert [message["content"] for message in retained] == ["s", "aa", "bb", "cc"]
    event = manager.events[0]
    assert event["original_count"] == 5
    assert event["retained_count"] == 4
    assert event["dropped_count"] == 1
    assert event["original_content_bytes"] == 12
    assert event["retained_content_bytes"] == 7
    assert event["strategy"] == "truncation"


def test_oversized_system_or_protected_messages_raise():
    for messages, limit in [
        ([{"role": "system", "content": "system"}, {"role": "user", "content": "x"}], 5),
        (
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "offer!"},
                {"role": "assistant", "content": "bad!!!"},
                {"role": "user", "content": "correct"},
            ],
            10,
        ),
    ]:
        manager = SlidingWindow(max_messages=4, max_content_bytes=limit)

        try:
            manager(messages)
        except ContextLimitError:
            pass
        else:
            raise AssertionError("an oversized protected context must be rejected")

        assert manager.events[-1]["status"] == "rejected"
        assert manager.events[-1]["strategy"] == "truncation"


def test_latest_offer_malformed_reply_and_correction_survive():
    manager = SlidingWindow(max_messages=4, max_content_bytes=3000)
    messages = [
        {"role": "system", "content": "negotiation rules"},
        {"role": "user", "content": "old bargaining detail"},
        {"role": "assistant", "content": "old response"},
        {"role": "user", "content": "latest offer: 700 kr"},
        {"role": "assistant", "content": "malformed reply"},
        {"role": "user", "content": "correction request"},
    ]

    retained = manager(messages)

    assert [message["content"] for message in retained[-3:]] == [
        "latest offer: 700 kr",
        "malformed reply",
        "correction request",
    ]


def test_context_manager_validation_rejects_invalid_limits_and_inputs():
    for keyword, value in [
        ("max_messages", True),
        ("max_messages", 2),
        ("max_messages", 4.0),
        ("max_content_bytes", True),
        ("max_content_bytes", 0),
        ("max_content_bytes", 4.0),
    ]:
        try:
            SlidingWindow(**{keyword: value})
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid {keyword} should be rejected")

    manager = SlidingWindow()
    for invalid in ([], [{"role": "user", "content": "missing system"}]):
        try:
            manager(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid message input should be rejected")


def test_thirty_turn_engine_context_stress():
    agents = [
        Agent("Buyer", "Keep replies short."),
        Agent("Seller", "Keep replies short."),
    ]
    client = RecordingMockClient()
    manager = SlidingWindow(max_messages=4, max_content_bytes=3000)
    engine = DialogueEngine(
        agents,
        client,
        Budget(max_turns=30, max_tokens=100000, max_seconds=60),
        manage_context=manager,
    )

    transcript = engine.run()

    assert len(transcript) == 30
    assert engine.budget.stop_reason == "max_turns"
    assert all(entry.content.startswith("short reply ") for entry in transcript)
    assert len(client.requests) == 30
    assert all(len(request) <= 5 for request in client.requests)
    assert all(
        sum(len(message["content"].encode("utf-8")) for message in request)
        <= manager.max_content_bytes
        for request in client.requests
    )
    assert manager.events
    assert all(event["strategy"] == "truncation" for event in manager.events)

    evidence_path = Path("transcripts") / f"context_stress_mock_{uuid4().hex}.json"
    engine.save(
        evidence_path,
        meta={
            "labels": ["mock", "stress-test"],
            "context_settings": {
                "max_messages": manager.max_messages,
                "max_content_bytes": manager.max_content_bytes,
            },
            "context_events": manager.events,
        },
    )

    with evidence_path.open() as file:
        evidence = json.load(file)
    assert len(evidence["messages"]) == 30
    assert evidence["meta"]["context_events"] == manager.events
    assert evidence["meta"]["labels"] == ["mock", "stress-test"]
    assert evidence["meta"]["context_settings"] == {
        "max_messages": 4,
        "max_content_bytes": 3000,
    }

    print(f"Evidence path: {evidence_path}")
    print("Context-management checks passed.")


def test_engine_stops_before_rejected_context_request():
    agents = [
        Agent("Buyer", "small"),
        Agent("Seller", "S" * 100),
    ]
    client = RecordingMockClient()
    manager = SlidingWindow(max_messages=4, max_content_bytes=30)
    engine = DialogueEngine(
        agents,
        client,
        Budget(max_turns=30, max_tokens=100000, max_seconds=60),
        manage_context=manager,
    )

    transcript = engine.run()

    assert len(client.requests) == 1
    assert len(transcript) == 1
    assert transcript[0].content == "short reply 1"
    assert engine.budget.turns == 1
    assert engine.budget.tokens == (
        transcript[0].prompt_tokens + transcript[0].completion_tokens
    )
    assert engine.budget.stop_reason == "context_limit"
    assert manager.events[-1]["status"] == "rejected"


if __name__ == "__main__":
    test_short_input_is_unchanged_without_mutation()
    test_count_limit_keeps_recent_messages_in_order_and_roles()
    test_byte_limit_removes_old_messages_before_count_limit()
    test_oversized_system_or_protected_messages_raise()
    test_latest_offer_malformed_reply_and_correction_survive()
    test_context_manager_validation_rejects_invalid_limits_and_inputs()
    test_thirty_turn_engine_context_stress()
    test_engine_stops_before_rejected_context_request()
