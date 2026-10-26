"""run.py: entry point.

Two modes:
  1. Smoke test (default): one model call, to prove your setup works.
        python run.py --mock      # free/offline
        python run.py             # real local model via Ollama
  2. Full run from a config (works once you've completed engine.py in Week 2):
        python run.py --config configs/debate.yaml
        python run.py --config configs/debate.yaml --mock
"""
import argparse
import copy
import uuid
from datetime import datetime, timezone
from pathlib import Path

from llm_client import MockClient, make_client


def smoke(mock):
    client = make_client(mock=mock)
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Say hello in exactly three words."},
    ]
    r = client.chat("llama3.2:3b", messages, temperature=0)
    print("Model replied:", r.text)
    print(f"(prompt={r.prompt_tokens} tokens, completion={r.completion_tokens} tokens, "
          f"{r.seconds:.2f}s)")


def run_config(path, mock):
    import yaml  # local import so the smoke test needs no extra deps

    from agents import Agent
    from budget import Budget
    from context_manager import SlidingWindow
    from engine import DialogueEngine
    from negotiation import agreement_reached, negotiation_stop_reason, parse_action

    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Configuration-driven execution was implemented with AI assistance.
    agents = []
    for agent_config in cfg["agents"]:
        agent_settings = dict(agent_config)
        if cfg.get("topic"):
            agent_settings["system_prompt"] = (
                f'{agent_settings["system_prompt"]}\nTask: {cfg["topic"]}'
            )
        agents.append(Agent(**agent_settings))
    budget = Budget(**cfg.get("budget", {}))
    if mock and cfg.get("mock_replies") is not None:
        client = MockClient(replies=cfg["mock_replies"])
    else:
        client = make_client(mock=mock, options=cfg.get("model_options"))

    engine_options = {}
    context_manager = None
    context_config = cfg.get("context")
    if context_config:
        if context_config.get("strategy") != "truncation":
            raise ValueError("only the truncation context strategy is supported")
        context_manager = SlidingWindow(
            max_messages=context_config["max_messages"],
            max_content_bytes=context_config["max_content_bytes"],
        )
        engine_options["manage_context"] = context_manager
    negotiation = cfg.get("negotiation")
    if negotiation:
        # Structured negotiation hooks were wired with AI assistance.
        engine_options["goal_reached"] = lambda transcript: agreement_reached(
            transcript,
            negotiation["buyer_name"],
            negotiation["seller_name"],
            negotiation["buyer_max"],
            negotiation["seller_min"],
        )
        engine_options["reply_validator"] = parse_action
        engine_options["max_parse_retries"] = negotiation.get(
            "max_parse_retries", 1
        )
        engine_options["stop_condition"] = negotiation_stop_reason
    engine = DialogueEngine(agents, client, budget, **engine_options)
    run_started_at = datetime.now(timezone.utc).isoformat()
    transcript = engine.run()
    valid_history = [entry for entry in transcript if entry.parse_valid]

    for e in transcript:
        print(f"{e.speaker}: {e.content}\n")

    configured_output = Path(cfg.get("output", "transcripts/run.json"))
    mode = "mock" if mock else "real"
    output_name = (
        f"{configured_output.stem}_{mode}_{uuid.uuid4().hex}"
        f"{configured_output.suffix or '.json'}"
    )
    out = configured_output.with_name(output_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "topic": cfg.get("topic"),
        "config_path": path,
        "config": copy.deepcopy(cfg),
        "mock_mode": bool(mock),
        "run_started_at": run_started_at,
    }
    if negotiation:
        agreement_success = agreement_reached(
            valid_history,
            negotiation["buyer_name"],
            negotiation["seller_name"],
            negotiation["buyer_max"],
            negotiation["seller_min"],
        )
        agreed_price = None
        if agreement_success:
            agreed_price = parse_action(valid_history[-1].content)["price"]
        metadata["negotiation_outcome"] = {
            "success": agreement_success,
            "agreed_price": agreed_price,
        }
    if context_manager is not None:
        metadata["context_settings"] = copy.deepcopy(context_config)
        metadata["context_events"] = copy.deepcopy(context_manager.events)
    engine.save(out, meta=metadata)
    print(f"[stopped: {budget.stop_reason} "
          f"({budget.turns} turns, {budget.tokens} tokens). Saved {out}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true", help="use the free offline MockClient")
    p.add_argument("--config", help="run a full dialogue from a YAML config")
    args = p.parse_args()
    if args.config:
        run_config(args.config, mock=args.mock)
    else:
        smoke(mock=args.mock)
