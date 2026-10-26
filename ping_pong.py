"""ping_pong.py: WEEK 1 reference: the simplest possible two-agent loop.

This works out of the box. It uses NAIVE role handling: each agent is shown the
whole conversation as a single 'user' message. That's fine for Week 1; the point
is to see two agents take turns under a Budget. In Week 2 you replace this with
engine.py and PROPER role-mapping (each agent sees the other as 'user', itself as
'assistant').

Run:  python ping_pong.py --mock          # free, instant
      python ping_pong.py                  # real local model
      python ping_pong.py --turns 6
"""
import argparse

from agents import Agent
from budget import Budget
from llm_client import make_client

# === Your two personas. EDIT these for your chosen scenario. ===
# Persona prompts drafted with ChatGPT assistance and applied using Copilot.
TOPIC = "Negotiate the price of a used bicycle in Norwegian kroner (kr)."
AGENT_A = Agent(
    "Buyer",
    "Always reply in English. You are buying a used bicycle. Aim for the lowest price you can negotiate. "
    "Your maximum purchase price is 800 kr; never offer or accept more. Start by "
    "offering 500 kr. Respond to the seller's latest message. Use whole-number "
    "prices in kr. Make one clear offer or acceptance per message. Accept only "
    "the seller's latest offer and repeat its exact price when accepting. You may "
    "explicitly declare NO DEAL if you cannot agree. Do not invent the seller's "
    "response or claim acceptance on their behalf. Keep each reply to at most two "
    "sentences.",
    model="llama3.2:3b",
    temperature=0.7,
)
AGENT_B = Agent(
    "Seller",
    "Always reply in English. You are selling a used bicycle. Aim for the highest price you can negotiate. "
    "Your minimum sale price is 600 kr; never offer or accept less. Start by asking "
    "for 900 kr unless you choose to accept the buyer's offer. Respond to the buyer's "
    "latest message. Use whole-number prices in kr. Make one clear offer or acceptance "
    "per message. Accept only the buyer's latest offer and repeat its exact price when "
    "accepting. You may explicitly declare NO DEAL if you cannot agree. Do not invent "
    "the buyer's response or claim acceptance on their behalf. Keep each reply to at "
    "most two sentences.",
    model="llama3.2:3b",
    temperature=0.7,
)


def render(transcript):
    if not transcript:
        return "(You speak first.)"
    return "\n".join(f"{name}: {text}" for name, text in transcript)


def main(mock, turns):
    client = make_client(mock=mock)
    agents = [AGENT_A, AGENT_B]
    transcript = []  # list of (name, text)

    # The Budget is what makes this safe. Never write this loop without one.
    budget = Budget(max_turns=turns, max_tokens=4000, max_seconds=120)

    while not budget.exhausted():
        speaker = agents[len(transcript) % 2]
        messages = [
            {"role": "system",
             "content": f"{speaker.system_prompt}\nNegotiation task: {TOPIC}"},
            {"role": "user",
             "content": f"Conversation so far:\n{render(transcript)}\n\n"
                        f"Your turn, {speaker.name}. Reply with one short message."},
        ]
        reply = client.chat(speaker.model, messages, temperature=speaker.temperature)
        transcript.append((speaker.name, reply.text))
        budget.record(turns=1, tokens=reply.tokens)
        print(f"{speaker.name}: {reply.text}\n")

    print(f"[stopped: {budget.stop_reason} "
          f"({budget.turns} turns, {budget.tokens} tokens, {budget.elapsed:.1f}s)]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true", help="use the free offline MockClient")
    p.add_argument("--turns", type=int, default=8, help="hard cap on number of turns")
    args = p.parse_args()
    main(mock=args.mock, turns=args.turns)
