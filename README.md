# AI Bicycle Negotiation

An educational multi-agent negotiation project in which a buyer and seller
agent negotiate the price of a used bicycle. The project demonstrates
role-aware dialogue orchestration, bounded model calls, structured negotiation
actions, transcript recording, and a separate model-based judge.

The implementation supports a free offline mock client and a real local
Ollama client. It is intended as a student portfolio project, not as a
production negotiation system.

## Features

- Buyer and seller personas with configurable prompts, models, and temperatures.
- Role-aware conversation views: each agent sees its own previous messages as
  `assistant` messages and the other agent's messages as `user` messages.
- Safety budgets for maximum turns, tokens, and elapsed time.
- Optional sliding-window context management with a protected recent history.
- Structured negotiation actions: `offer`, `accept`, and `no_deal`.
- Validation and bounded retries for malformed model responses.
- Deterministic agreement checks using buyer and seller price limits.
- JSON transcript persistence with token totals, timing, parse status, and stop
  reasons.
- A separate judge helper that validates a JSON verdict and records attempts.
- Mocked HTTP and offline tests for the core engine, client, budget, context,
  negotiation, and judge behavior.

## Architecture

1. [`agents.py`](agents.py) defines the `Agent` data object: name, system
   prompt, model, and temperature.
2. [`llm_client.py`](llm_client.py) provides two compatible clients:
   `MockClient` returns canned replies without a model server, while
   `OllamaClient` sends chat requests to Ollama's local `/api/chat` endpoint.
3. [`engine.py`](engine.py) contains `DialogueEngine`. It selects speakers in
   order, builds each speaker's role-aware context, applies context management,
   calls the client, records usage, validates replies, and stops on a budget,
   goal, parse, timeout, or domain condition.
4. [`negotiation.py`](negotiation.py) parses structured actions and checks
   whether the final offer and acceptance form a valid agreement within the
   configured price limits.
5. [`budget.py`](budget.py) provides independent turn, token, and time
   guardrails.
6. [`evaluate.py`](evaluate.py) loads a saved transcript and asks a separate
   configured judge model to assess it. The judge result is evidence for
   analysis; it does not replace the engine's deterministic agreement result.
7. [`judge.py`](judge.py) builds evidence-bound judge prompts, validates judge
   JSON, retries malformed responses, and records accounting.
8. [`run.py`](run.py) is the main smoke-test entry point and contains the
   configuration-driven runner for a YAML scenario.

[`ping_pong.py`](ping_pong.py) is an earlier, simpler two-agent reference loop.
[`statelessness.py`](statelessness.py) demonstrates the difference between
fresh model context and supplied conversation history.
[`make_results.py`](make_results.py) contains a manifest-based results workflow.

## Technologies and dependencies

- Python 3.10 or newer (the code uses modern union type syntax).
- [Requests](https://requests.readthedocs.io/) for Ollama HTTP calls and
  mocked HTTP tests.
- [PyYAML](https://pyyaml.org/) for the configuration-driven runner and
  evaluation scripts.
- Ollama for real local model calls; it is not a Python dependency.
- No hosted API provider is used by the current implementation.

Install the Python dependencies from [`requirements.txt`](requirements.txt):

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Configuration and model requirements

The offline mock mode does not require Ollama or network access:

```bat
python run.py --mock
```

This command was verified against the current source and performs a one-call
smoke test using `MockClient`.

Real model calls require:

1. Ollama installed and running locally.
2. The model named by the call available in Ollama. The current source uses
   `llama3.2:3b` by default.
3. Ollama's default API endpoint at `http://localhost:11434`, unless a caller
   constructs `OllamaClient` with another host.

For example:

```bat
ollama pull llama3.2:3b
ollama serve
```

Keep `ollama serve` running in one terminal and run the Python command in
another. The default `python run.py` command performs a real one-call smoke
test and therefore requires the local model server.

The repository currently does not contain the `configs/` directory referenced
by the configuration-driven code and older documentation. Consequently, a
full YAML-configured negotiation cannot be run from this checkout until a
compatible YAML file is supplied. A configuration must define `agents`,
optionally `topic`, `budget`, `context`, `negotiation`, `model_options`,
`mock_replies`, and `output` as used by [`run.py`](run.py).

## Running negotiations

The verified offline smoke command is:

```bat
python run.py --mock
```

The earlier reference loop can also be run offline:

```bat
python ping_pong.py --mock --turns 6
```

To run a full scenario when you have a compatible YAML configuration:

```bat
python run.py --config path\to\negotiation.yaml --mock
```

Omit `--mock` to call Ollama. Full runs save a JSON transcript under the
configured output directory and include the mode and a UUID in the filename.

## Tests and evaluations

The tests are standalone Python scripts. Run them from this directory:

```bat
python test_budgets.py
python test_client_deadline.py
python test_client_options.py
python test_client_timeout.py
python test_context.py
python test_engine.py
python test_engine_timeout.py
python test_judge.py
python test_negotiation.py
python test_parse_guard.py
```

They use mocks or mocked HTTP and do not require Ollama. The test scripts
write generated evidence under `transcripts/`, which is ignored by Git.

[`evaluate.py`](evaluate.py) requires both a saved transcript and a judge YAML
configuration. It always uses a real Ollama client in the current code:

```bat
python evaluate.py --transcript path\to\transcript.json --config path\to\judge.yaml
```

[`make_results.py`](make_results.py) requires a manifest at
`configs/experiment_runs.yaml`, the referenced transcript/evaluation JSON
files, and the associated output directories. Those inputs are not present in
this checkout, so no results command can be verified from the current folder.

## What the project demonstrates

The project demonstrates the mechanics of bounded, role-aware local-LLM
dialogue and deterministic validation of a structured negotiation. It does not
provide a benchmark of negotiation quality. Outcomes depend on the model,
prompt, temperature, budgets, and input configuration. A run can stop at the
turn, token, or time limit, after a timeout or parse failure, with `no_deal`,
or after a valid agreement. The judge can also reject malformed output or
return a verdict that conflicts with the deterministic engine result; judge
explanations should therefore be treated as analysis evidence rather than
ground truth.

No experiment results are included in this checkout, and no successful or
unsuccessful benchmark outcome should be inferred beyond what a particular
saved transcript records.

## Known limitations

- The full scenario and evaluation YAML files referenced by the source are not
  included, so the configuration-driven workflow is incomplete in this
  checkout.
- Real runs require a locally installed Ollama model and can be slow or
  resource-intensive.
- The caller timeout bounds waiting, but an underlying HTTP or model operation
  may continue in the background.
- Token counts from `MockClient` are word-count approximations.
- Agreement validation is intentionally narrow: it checks the final
  offer/acceptance pair and configured price limits rather than understanding
  natural-language intent.
- The judge is another model call and is not an authoritative evaluator.
- Generated transcripts and evaluation outputs are local artifacts and are not
  part of the source tree.

## Attribution

This is an individual student project extending course starter code. AI tools
were used for assistance with requirements understanding, planning, code
review, testing, and documentation. The project author remains responsible
for reviewing and understanding the implementation.
