"""engine.py: WEEK 2 target: a reusable two-agent dialogue engine.

The guardrails (Budget) and the bookkeeping (Entry, save) are DONE.
You implement the two parts marked `# TODO (WEEK 2)`:
  1. view_for(...)      - render the conversation from one agent's point of view
  2. DialogueEngine.run - the orchestration loop

In WEEK 3 you also implement the `manage_context` hook (truncation/summarisation).

Read ../handouts/week2-dialogue-engine.md alongside this file.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from budget import Budget
from context_manager import ContextLimitError


@dataclass
class Entry:
    """One message in the conversation, with its measured cost."""
    speaker: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    turn_index: int
    parse_valid: bool = True
    parse_error: str | None = None


def view_for(agent, transcript):
    """Build the message list to send to `agent`, from ITS point of view.

    Requirements (this is the key idea of Week 2):
      - first message: {"role": "system", "content": agent.system_prompt}
      - for each Entry in transcript:
            role = "assistant" if it was spoken by THIS agent
            role = "user"      if it was spoken by the OTHER agent
      - return the list of {"role", "content"} dicts.

    Tip: test it by hand-building a 3-entry transcript and printing the view for
    each of your two agents; the roles should be mirror images.
    """
    # Implemented with AI assistance and reviewed for the Week 2 requirements.
    messages = [{"role": "system", "content": agent.system_prompt}]
    for entry in transcript:
        role = "assistant" if entry.speaker == agent.name else "user"
        messages.append({"role": role, "content": entry.content})
    if not transcript:
        messages.append({"role": "user", "content": "You speak first."})
    return messages


class DialogueEngine:
    """Runs two (or more) agents in turn until a Budget stop fires.

    agents:        list of Agent; turn order follows the list, then wraps.
    client:        an OllamaClient or MockClient.
    budget:        a Budget instance (your three guardrails).
    goal_reached:  optional fn(transcript) -> bool; return True when the
                   scenario goal is met (engine then stops with 'goal_reached').
    manage_context: optional fn(messages) -> messages; WEEK 3 hook to keep the
                   message list inside the context window. Default: no-op.
    """

    def __init__(
        self,
        agents,
        client,
        budget,
        goal_reached=None,
        manage_context=None,
        reply_validator=None,
        max_parse_retries=1,
        stop_condition=None,
    ):
        if (
            isinstance(max_parse_retries, bool)
            or not isinstance(max_parse_retries, int)
            or max_parse_retries < 0
        ):
            raise ValueError("max_parse_retries must be a nonnegative integer")
        self.agents = agents
        self.client = client
        self.budget = budget
        self.transcript = []  # list[Entry]
        self.goal_reached = goal_reached or (lambda t: False)
        self.manage_context = manage_context or (lambda messages: messages)
        self.reply_validator = reply_validator
        self.max_parse_retries = max_parse_retries
        self.stop_condition = stop_condition or (lambda transcript: None)
        self.request_errors = []

    def next_speaker(self):
        valid_turns = sum(entry.parse_valid for entry in self.transcript)
        return self.agents[valid_turns % len(self.agents)]

    def run(self):
        """Run the dialogue to completion. Return the final transcript.

        Execution turns count every generated reply, including correction
        attempts after malformed replies.

        Loop shape (fill in the body):
            while not self.budget.exhausted():
                speaker = self.next_speaker()
                messages = self.manage_context(view_for(speaker, self.transcript))
                remaining_seconds = budget.max_seconds - budget.elapsed
                reply = self.client.chat(
                    speaker.model,
                    messages,
                    speaker.temperature,
                    timeout=min(300, remaining_seconds),
                )
                append an Entry(...) to self.transcript
                self.budget.record(turns=1, tokens=reply.tokens)
                if self.goal_reached(self.transcript):
                    self.budget.stop("goal_reached")
            return self.transcript
        """
        # Parse retries are bounded, budgeted correction attempts implemented with AI assistance.
        retry_speaker = None
        retry_count = 0
        while not self.budget.exhausted():
            valid_history = [
                entry for entry in self.transcript if entry.parse_valid
            ]
            speaker = retry_speaker or self.next_speaker()
            messages = view_for(speaker, valid_history)
            if retry_speaker is not None:
                previous = self.transcript[-1]
                messages.extend([
                    {"role": "assistant", "content": previous.content},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was invalid: {previous.parse_error}. "
                            "Return the required JSON object only."
                        ),
                    },
                ])
            try:
                messages = self.manage_context(messages)
            except ContextLimitError:
                self.budget.stop("context_limit")
                break
            remaining_seconds = self.budget.max_seconds - self.budget.elapsed
            if remaining_seconds <= 0:
                self.budget.stop("max_seconds")
                break
            timeout = min(300, remaining_seconds)
            attempted_turn_index = len(self.transcript)
            request_started = self.budget.elapsed
            try:
                reply = self.client.chat(
                    speaker.model,
                    messages,
                    temperature=speaker.temperature,
                    timeout=timeout,
                )
            except requests.Timeout as exc:
                request_duration = self.budget.elapsed - request_started
                self.request_errors.append(
                    {
                        "speaker": speaker.name,
                        "turn_index": attempted_turn_index,
                        "timeout": timeout,
                        "request_duration": request_duration,
                        "exception_type": type(exc).__name__,
                        "explanation": (
                            "Token usage for the unfinished request is unknown."
                        ),
                    }
                )
                if self.budget.elapsed >= self.budget.max_seconds:
                    self.budget.stop("max_seconds")
                else:
                    self.budget.stop("request_timeout")
                break
            turn_index = len(self.transcript)
            parse_valid = True
            parse_error = None
            if self.reply_validator is not None:
                try:
                    self.reply_validator(reply.text)
                except ValueError as exc:
                    parse_valid = False
                    parse_error = str(exc)
            self.transcript.append(
                Entry(
                    speaker=speaker.name,
                    content=reply.text,
                    prompt_tokens=reply.prompt_tokens,
                    completion_tokens=reply.completion_tokens,
                    seconds=reply.seconds,
                    turn_index=turn_index,
                    parse_valid=parse_valid,
                    parse_error=parse_error,
                )
            )
            self.budget.record(turns=1, tokens=reply.tokens)
            if self.budget.elapsed >= self.budget.max_seconds:
                self.budget.stop("max_seconds")
                break
            if not parse_valid:
                if self.budget.exhausted():
                    break
                if retry_count < self.max_parse_retries:
                    retry_speaker = speaker
                    retry_count += 1
                    continue
                self.budget.stop("parse_error")
                break

            retry_speaker = None
            retry_count = 0
            valid_history = [
                entry for entry in self.transcript if entry.parse_valid
            ]
            if self.goal_reached(valid_history):
                self.budget.stop("goal_reached")
            else:
                stop_reason = self.stop_condition(valid_history)
                if stop_reason:
                    self.budget.stop(stop_reason)
        return self.transcript

    # === bookkeeping below is DONE ===

    def save(self, path, meta=None):
        """Write a structured JSON transcript: a run header + every message."""
        record = {
            "meta": meta or {},
            "agents": [
                {"name": a.name, "model": a.model, "temperature": a.temperature}
                for a in self.agents
            ],
            "budget": self.budget.summary(),
            "totals": {
                "turns": len(self.transcript),
                "prompt_tokens": sum(e.prompt_tokens for e in self.transcript),
                "completion_tokens": sum(e.completion_tokens for e in self.transcript),
                "seconds": round(sum(e.seconds for e in self.transcript), 3),
            },
            "request_errors": self.request_errors,
            "messages": [asdict(e) for e in self.transcript],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        return path
