import json


def parse_action(text):
    """Validate and return one negotiation action; implemented with AI assistance."""
    try:
        action = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Malformed JSON: {exc}") from exc

    if not isinstance(action, dict):
        raise ValueError("Action must be one JSON object")
    for field in ("action", "price", "message"):
        if field not in action:
            raise ValueError(f"Missing required field: {field}")

    if not isinstance(action["action"], str):
        raise ValueError("action must be a string")
    if action["action"] not in {"offer", "accept", "no_deal"}:
        raise ValueError("action must be offer, accept, or no_deal")
    if not isinstance(action["message"], str):
        raise ValueError("message must be a string")

    if action["action"] == "no_deal":
        if action["price"] is not None:
            raise ValueError("no_deal price must be null")
    elif (
        isinstance(action["price"], bool)
        or not isinstance(action["price"], int)
        or action["price"] <= 0
    ):
        raise ValueError("offer and accept price must be a positive integer")

    return action


def agreement_reached(transcript, buyer_name, seller_name, buyer_max, seller_min):
    """Return whether the final offer and acceptance form a valid agreement."""
    if len(transcript) < 2:
        return False

    earlier, latest = transcript[-2:]
    if earlier.speaker == latest.speaker:
        return False
    if {earlier.speaker, latest.speaker} != {buyer_name, seller_name}:
        return False

    try:
        offer = parse_action(earlier.content)
        acceptance = parse_action(latest.content)
    except ValueError:
        return False

    return (
        offer["action"] == "offer"
        and acceptance["action"] == "accept"
        and offer["price"] == acceptance["price"]
        and seller_min <= offer["price"] <= buyer_max
    )


def negotiation_stop_reason(transcript):
    """Return no_deal for a valid final no-deal action; added with AI assistance."""
    if not transcript:
        return None
    try:
        action = parse_action(transcript[-1].content)
    except ValueError:
        return None
    return "no_deal" if action["action"] == "no_deal" else None