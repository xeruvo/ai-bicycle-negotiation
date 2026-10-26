import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from agents import Agent
from budget import Budget
from engine import DialogueEngine
from llm_client import MockClient


# All results in this file are mock development tests, not real-model experiments.
BUYER = Agent("Buyer", "Buyer mock development prompt.")
SELLER = Agent("Seller", "Seller mock development prompt.")
TRANSCRIPTS_DIR = Path("transcripts")


class DelayedMockClient(MockClient):
    def chat(self, model, messages, temperature=0.7, timeout=300):
        started = time.monotonic()
        time.sleep(0.1)
        reply = super().chat(model, messages, temperature, timeout)
        reply.seconds = time.monotonic() - started
        return reply


def save_and_check(engine, test_name, timing_metadata):
    saved_path = TRANSCRIPTS_DIR / f"{test_name}_{uuid.uuid4().hex}.json"
    metadata = {
        "mode": "mock development test",
        "mock_mode": True,
        "test_name": test_name,
        "timing": timing_metadata,
    }
    engine.save(saved_path, meta=metadata)
    with saved_path.open() as file:
        saved = json.load(file)

    assert saved["meta"] == metadata
    assert saved["messages"] == [asdict(entry) for entry in engine.transcript]
    assert saved["agents"] == [
        {
            "name": agent.name,
            "model": agent.model,
            "temperature": agent.temperature,
        }
        for agent in engine.agents
    ]
    assert saved["budget"]["limits"] == {
        "max_turns": engine.budget.max_turns,
        "max_tokens": engine.budget.max_tokens,
        "max_seconds": engine.budget.max_seconds,
    }
    assert saved["budget"]["stop_reason"] == engine.budget.stop_reason
    assert saved["totals"]["turns"] == len(engine.transcript)
    assert saved["totals"]["prompt_tokens"] == sum(
        entry.prompt_tokens for entry in engine.transcript
    )
    assert saved["totals"]["completion_tokens"] == sum(
        entry.completion_tokens for entry in engine.transcript
    )
    assert saved["totals"]["seconds"] == round(
        sum(entry.seconds for entry in engine.transcript), 3
    )
    return saved_path


def run_turn_test():
    test_name = "turn_budget_mock"
    budget = Budget(max_turns=2, max_tokens=4000, max_seconds=60)
    engine = DialogueEngine([BUYER, SELLER], MockClient(), budget)
    engine.run()

    assert len(engine.transcript) == 2
    assert budget.stop_reason == "max_turns"
    assert budget.tokens < budget.max_tokens
    assert budget.elapsed < budget.max_seconds
    saved_path = save_and_check(engine, test_name, "mock timing without artificial delay")
    print(test_name, budget.stop_reason, len(engine.transcript), saved_path)


def run_token_test():
    test_name = "token_budget_mock"
    budget = Budget(max_turns=20, max_tokens=1, max_seconds=60)
    engine = DialogueEngine([BUYER, SELLER], MockClient(), budget)
    engine.run()

    # The completed call can exceed the threshold because tokens are counted after the reply.
    assert len(engine.transcript) == 1
    assert budget.stop_reason == "max_tokens"
    assert budget.turns < budget.max_turns
    assert budget.elapsed < budget.max_seconds
    saved_path = save_and_check(engine, test_name, "mock timing without artificial delay")
    print(test_name, budget.stop_reason, len(engine.transcript), saved_path)


def run_time_test():
    test_name = "time_budget_mock"
    budget = Budget(max_turns=20, max_tokens=4000, max_seconds=0.05)
    engine = DialogueEngine([BUYER, SELLER], DelayedMockClient(), budget)
    engine.run()

    # This tests stopping between calls, not interrupting a call in progress.
    assert len(engine.transcript) == 1
    assert budget.stop_reason == "max_seconds"
    assert budget.turns < budget.max_turns
    assert budget.tokens < budget.max_tokens
    saved_path = save_and_check(
        engine,
        test_name,
        "measured with an artificial delay",
    )
    print(test_name, budget.stop_reason, len(engine.transcript), saved_path)


run_turn_test()
run_token_test()
run_time_test()
print("All budget and JSON checks passed.")