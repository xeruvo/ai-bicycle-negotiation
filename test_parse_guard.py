import json
import uuid
from dataclasses import asdict
from pathlib import Path

from agents import Agent
from budget import Budget
from engine import DialogueEngine
from llm_client import MockClient
from negotiation import agreement_reached, parse_action


# Mock-only development checks; no real-model experiment is performed here.
BUYER = Agent("Buyer", "Buyer parse-guard test prompt.")
SELLER = Agent("Seller", "Seller parse-guard test prompt.")
TRANSCRIPTS_DIR = Path("transcripts")


class RecordingMockClient(MockClient):
    def __init__(self, replies):
        super().__init__(replies=replies)
        self.requests = []

    def chat(self, model, messages, temperature=0.7, timeout=300):
        self.requests.append([dict(message) for message in messages])
        return super().chat(model, messages, temperature, timeout)


def goal_reached(transcript):
    return agreement_reached(transcript, "Buyer", "Seller", 800, 600)


def token_total(engine):
    return sum(
        entry.prompt_tokens + entry.completion_tokens
        for entry in engine.transcript
    )


valid_offer = '{"action": "offer", "price": 700, "message": "I offer 700 kr."}'
valid_accept = '{"action": "accept", "price": 700, "message": "I accept 700 kr."}'
malformed = '{"action": "offer", "price": 700, '

recovery_client = RecordingMockClient([valid_offer, malformed, valid_accept])
recovery_engine = DialogueEngine(
    [BUYER, SELLER],
    recovery_client,
    Budget(max_turns=6, max_tokens=4000, max_seconds=60),
    goal_reached=goal_reached,
    reply_validator=parse_action,
    max_parse_retries=1,
)
recovery_transcript = recovery_engine.run()
assert [entry.speaker for entry in recovery_transcript] == [
    "Buyer",
    "Seller",
    "Seller",
]
assert len(recovery_transcript) == 3
assert [entry.parse_valid for entry in recovery_transcript] == [True, False, True]
assert recovery_transcript[1].parse_error
assert recovery_engine.budget.stop_reason == "goal_reached"
assert token_total(recovery_engine) == recovery_engine.budget.tokens
assert malformed in [
    message["content"]
    for message in recovery_client.requests[2]
]
assert any(
    message["role"] == "user" and "invalid" in message["content"]
    for message in recovery_client.requests[2]
)

evidence_path = TRANSCRIPTS_DIR / f"parse_guard_recovery_mock_{uuid.uuid4().hex}.json"
recovery_engine.save(evidence_path, meta={
    "mode": "mock development test",
    "test_name": "parse_guard_recovery",
})
with evidence_path.open() as file:
    saved_recovery = json.load(file)
assert saved_recovery["messages"] == [
    asdict(entry) for entry in recovery_transcript
]
assert saved_recovery["budget"]["stop_reason"] == "goal_reached"
assert saved_recovery["budget"]["tokens"] == recovery_engine.budget.tokens
assert saved_recovery["totals"]["turns"] == 3
assert saved_recovery["totals"]["prompt_tokens"] == sum(
    entry.prompt_tokens for entry in recovery_transcript
)
assert saved_recovery["totals"]["completion_tokens"] == sum(
    entry.completion_tokens for entry in recovery_transcript
)
assert saved_recovery["totals"]["seconds"] == round(
    sum(entry.seconds for entry in recovery_transcript), 3
)

exhaustion_engine = DialogueEngine(
    [BUYER, SELLER],
    RecordingMockClient([malformed, malformed]),
    Budget(max_turns=6, max_tokens=4000, max_seconds=60),
    reply_validator=parse_action,
    max_parse_retries=1,
)
exhaustion_transcript = exhaustion_engine.run()
assert len(exhaustion_transcript) == 2
assert all(not entry.parse_valid for entry in exhaustion_transcript)
assert exhaustion_engine.budget.stop_reason == "parse_error"
assert token_total(exhaustion_engine) == exhaustion_engine.budget.tokens

no_retry_engine = DialogueEngine(
    [BUYER, SELLER],
    RecordingMockClient([malformed]),
    Budget(max_turns=6, max_tokens=4000, max_seconds=60),
    reply_validator=parse_action,
    max_parse_retries=0,
)
no_retry_transcript = no_retry_engine.run()
assert len(no_retry_transcript) == 1
assert not no_retry_transcript[0].parse_valid
assert no_retry_engine.budget.stop_reason == "parse_error"
assert token_total(no_retry_engine) == no_retry_engine.budget.tokens

budget_engine = DialogueEngine(
    [BUYER, SELLER],
    RecordingMockClient([malformed]),
    Budget(max_turns=1, max_tokens=4000, max_seconds=60),
    reply_validator=parse_action,
    max_parse_retries=1,
)
budget_transcript = budget_engine.run()
assert len(budget_transcript) == 1
assert not budget_transcript[0].parse_valid
assert budget_engine.budget.stop_reason == "max_turns"
assert token_total(budget_engine) == budget_engine.budget.tokens

print("Evidence path:", evidence_path)
print("Parse guard checks passed.")