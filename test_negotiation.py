import json

from agents import Agent
from budget import Budget
from engine import DialogueEngine, Entry
from llm_client import MockClient
from negotiation import (
    agreement_reached,
    negotiation_stop_reason,
    parse_action,
)


# Explicitly labelled dummy metrics for validation tests; not measurements.
DUMMY_METRICS = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "seconds": 0.0,
}
BUYER = "Buyer"
SELLER = "Seller"
buyer = Agent(BUYER, "Buyer integration-test prompt.")
seller = Agent(SELLER, "Seller integration-test prompt.")


def entry(speaker, content, turn_index):
    return Entry(speaker, content, **DUMMY_METRICS, turn_index=turn_index)


def action(action_name, price, message):
    return '{"action": "' + action_name + '", "price": ' + (
        "null" if price is None else str(price)
    ) + ', "message": "' + message + '"}'


def history(offer_speaker, offer_price, accept_speaker, accept_price):
    return [
        entry(offer_speaker, action("offer", offer_price, "offer"), 0),
        entry(accept_speaker, action("accept", accept_price, "accept"), 1),
    ]


def assert_invalid_json(text):
    try:
        parse_action(text)
    except ValueError:
        return
    assert False, f"Expected ValueError for {text!r}"


for price in (600, 700, 800):
    assert agreement_reached(
        history(BUYER, price, SELLER, price), BUYER, SELLER, 800, 600
    )
    assert agreement_reached(
        history(SELLER, price, BUYER, price), BUYER, SELLER, 800, 600
    )

assert not agreement_reached(
    history(BUYER, 550, SELLER, 550), BUYER, SELLER, 800, 600
)
assert not agreement_reached(
    history(SELLER, 850, BUYER, 850), BUYER, SELLER, 800, 600
)
assert not agreement_reached(
    history(BUYER, 700, SELLER, 750), BUYER, SELLER, 800, 600
)
assert not agreement_reached(
    history(BUYER, 700, BUYER, 700), BUYER, SELLER, 800, 600
)
assert not agreement_reached([], BUYER, SELLER, 800, 600)
assert not agreement_reached(
    [entry(BUYER, action("offer", 700, "offer"), 0)],
    BUYER,
    SELLER,
    800,
    600,
)
assert not agreement_reached(
    history("Stranger", 700, SELLER, 700), BUYER, SELLER, 800, 600
)
assert not agreement_reached(
    [
        entry(BUYER, action("offer", 600, "first offer"), 0),
        entry(SELLER, action("offer", 700, "counteroffer"), 1),
        entry(BUYER, action("accept", 600, "accept older offer"), 2),
    ],
    BUYER,
    SELLER,
    800,
    600,
)
assert not agreement_reached(
    [
        entry(BUYER, action("offer", 700, "offer"), 0),
        entry(SELLER, action("no_deal", None, "no deal"), 1),
    ],
    BUYER,
    SELLER,
    800,
    600,
)
assert not agreement_reached(
    [
        entry(BUYER, action("offer", 700, "I accept this price"), 0),
        entry(SELLER, action("offer", 750, "counteroffer"), 1),
    ],
    BUYER,
    SELLER,
    800,
    600,
)
assert not agreement_reached(
    [
        entry(BUYER, action("offer", 700, "offer"), 0),
        entry(SELLER, action("offer", 700, "I accept"), 1),
    ],
    BUYER,
    SELLER,
    800,
    600,
)

valid_offer = {"action": "offer", "price": 700, "message": "offer"}
valid_accept = {"action": "accept", "price": 700, "message": "accept"}
for malformed_content, latest_content in (
    ("{", json.dumps(valid_accept)),
    (json.dumps(valid_offer), "{"),
    (
        json.dumps({"action": [], "price": 700, "message": "offer"}),
        json.dumps(valid_accept),
    ),
    (
        json.dumps(valid_offer),
        json.dumps({"action": {}, "price": 700, "message": "accept"}),
    ),
):
    assert not agreement_reached(
        [
            entry(BUYER, malformed_content, 0),
            entry(SELLER, latest_content, 1),
        ],
        BUYER,
        SELLER,
        800,
        600,
    )

assert_invalid_json("{")
assert_invalid_json("{}")
assert_invalid_json(action("withdraw", 700, "invalid action"))
assert_invalid_json("[]")
for invalid_action in ([], {}, None, True, 123):
    assert_invalid_json(json.dumps({
        "action": invalid_action,
        "price": 700,
        "message": "valid message",
    }))
assert_invalid_json(action("offer", '"700"', "string price"))
assert_invalid_json(action("offer", "700.0", "float price"))
assert_invalid_json(action("offer", "true", "boolean price"))
assert_invalid_json(action("offer", 0, "zero price"))
assert_invalid_json(action("offer", -1, "negative price"))
assert_invalid_json('{"action": "offer", "price": 700, "message": 1}')
assert_invalid_json(action("no_deal", 700, "non-null price"))
assert parse_action(action("no_deal", None, "no deal")) == {
    "action": "no_deal",
    "price": None,
    "message": "no deal",
}
assert negotiation_stop_reason([]) is None
assert negotiation_stop_reason([entry(BUYER, "{", 0)]) is None
assert negotiation_stop_reason([
    entry(BUYER, action("offer", 700, "NO DEAL"), 0),
]) is None
assert negotiation_stop_reason([
    entry(SELLER, action("no_deal", None, "I cannot agree."), 0),
]) == "no_deal"

print("Negotiation validation checks passed.")

valid_engine = DialogueEngine(
    [buyer, seller],
    MockClient(replies=[
        '{"action": "offer", "price": 700, "message": "I offer 700 kr."}',
        '{"action": "accept", "price": 700, "message": "I accept 700 kr."}',
    ]),
    Budget(max_turns=6, max_tokens=4000, max_seconds=60),
    goal_reached=lambda transcript: agreement_reached(
        transcript, BUYER, SELLER, 800, 600
    ),
)
valid_transcript = valid_engine.run()
assert len(valid_transcript) == 2
assert valid_engine.budget.stop_reason == "goal_reached"

invalid_engine = DialogueEngine(
    [buyer, seller],
    MockClient(replies=[
        '{"action": "offer", "price": 550, "message": "I offer 550 kr."}',
        '{"action": "accept", "price": 550, "message": "I accept 550 kr."}',
    ]),
    Budget(max_turns=2, max_tokens=4000, max_seconds=60),
    goal_reached=lambda transcript: agreement_reached(
        transcript, BUYER, SELLER, 800, 600
    ),
)
invalid_transcript = invalid_engine.run()
assert len(invalid_transcript) == 2
assert invalid_engine.budget.stop_reason == "max_turns"
assert not agreement_reached(invalid_transcript, BUYER, SELLER, 800, 600)

print("Negotiation engine integration checks passed.")

no_deal_engine = DialogueEngine(
    [buyer, seller],
    MockClient(replies=[
        '{"action": "offer", "price": 700, "message": "I offer 700 kr."}',
        '{"action": "no_deal", "price": null, "message": "I cannot agree."}',
    ]),
    Budget(max_turns=6, max_tokens=4000, max_seconds=60),
    goal_reached=lambda transcript: agreement_reached(
        transcript, BUYER, SELLER, 800, 600
    ),
    reply_validator=parse_action,
    stop_condition=negotiation_stop_reason,
)
no_deal_transcript = no_deal_engine.run()
assert len(no_deal_transcript) == 2
assert no_deal_engine.budget.stop_reason == "no_deal"
assert not agreement_reached(no_deal_transcript, BUYER, SELLER, 800, 600)

print("Negotiation stop-condition checks passed.")