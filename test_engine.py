from copy import deepcopy

from agents import Agent
from budget import Budget
from engine import DialogueEngine, Entry, view_for
from llm_client import MockClient


buyer = Agent("Buyer", "Buyer system prompt.")
seller = Agent("Seller", "Seller system prompt.")

# Explicitly labelled dummy metrics for this role-mapping test; not measurements.
DUMMY_METRICS = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "seconds": 0.0,
}
transcript = [
    Entry("Buyer", "I offer 500 kr.", **DUMMY_METRICS, turn_index=0),
    Entry("Seller", "I ask 900 kr.", **DUMMY_METRICS, turn_index=1),
    Entry("Buyer", "I offer 650 kr.", **DUMMY_METRICS, turn_index=2),
]
original_transcript = deepcopy(transcript)

buyer_view = view_for(buyer, transcript)
seller_view = view_for(seller, transcript)
print("Buyer view:", buyer_view)
print("Seller view:", seller_view)

assert buyer_view[0] == {"role": "system", "content": buyer.system_prompt}
assert seller_view[0] == {"role": "system", "content": seller.system_prompt}
assert [message["role"] for message in buyer_view[1:]] == [
    "assistant",
    "user",
    "assistant",
]
assert [message["role"] for message in seller_view[1:]] == [
    "user",
    "assistant",
    "user",
]
expected_content = [entry.content for entry in original_transcript]
assert [message["content"] for message in buyer_view[1:]] == expected_content
assert [message["content"] for message in seller_view[1:]] == expected_content
assert transcript == original_transcript

assert view_for(buyer, []) == [
    {"role": "system", "content": buyer.system_prompt},
    {"role": "user", "content": "You speak first."},
]
assert view_for(seller, []) == [
    {"role": "system", "content": seller.system_prompt},
    {"role": "user", "content": "You speak first."},
]

print("Role-mapping checks passed.")

engine = DialogueEngine(
    [buyer, seller],
    MockClient(),
    Budget(max_turns=6, max_tokens=4000, max_seconds=60),
)
engine_transcript = engine.run()

assert len(engine_transcript) == 6
assert [entry.speaker for entry in engine_transcript] == [
    "Buyer",
    "Seller",
    "Buyer",
    "Seller",
    "Buyer",
    "Seller",
]
assert [entry.turn_index for entry in engine_transcript] == list(range(6))
assert engine.budget.turns == 6
assert engine.budget.stop_reason == "max_turns"
assert engine.budget.tokens == sum(
    entry.prompt_tokens + entry.completion_tokens
    for entry in engine_transcript
)
assert all(entry.prompt_tokens > 0 for entry in engine_transcript)
assert all(entry.completion_tokens > 0 for entry in engine_transcript)
assert all(entry.seconds == 0.01 for entry in engine_transcript)

for entry in engine_transcript:
    print(f"{entry.speaker}: {entry.content}")
print(f"Stop reason: {engine.budget.stop_reason}")
print("Engine mock checks passed.")