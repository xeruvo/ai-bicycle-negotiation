"""Generate verified temperature experiment tables from an explicit manifest.

Prepared with AI assistance.
"""

from copy import deepcopy
import json
from pathlib import Path, PureWindowsPath

import yaml


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "configs" / "experiment_runs.yaml"
OUTPUT_PATH = ROOT / "docs" / "results.md"
EVALUATED_DIR = ROOT / "transcripts" / "evaluated"


def load_json(path):
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def resolved_path(root, relative_path):
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"manifest path must be relative: {relative_path}")
    return (root / path).resolve()


def normalized_negotiation_config(config):
    normalized = deepcopy(config)
    for agent in normalized.get("agents", []):
        if agent.get("name") == "Buyer":
            agent["temperature"] = "BUYER_TEMPERATURE"
    return normalized


def export_evaluated_transcripts(rows):
    """Export verified transcripts with their complete judge evidence."""
    # Evaluated-transcript export was prepared with AI assistance.
    EVALUATED_DIR.mkdir(parents=True, exist_ok=True)
    exported_paths = []
    for row in rows:
        evaluated_transcript = deepcopy(row["transcript_record"])
        evaluated_transcript["judge"] = deepcopy(row["evaluation_record"])
        evaluated_transcript["evidence_sources"] = {
            "transcript": row["transcript"],
            "evaluation": row["evaluation"],
        }
        output_path = EVALUATED_DIR / (
            f"{Path(row['transcript']).stem}_evaluated.json"
        )
        output_path.write_text(
            json.dumps(evaluated_transcript, indent=2), encoding="utf-8"
        )
        exported_paths.append(output_path)
    return exported_paths


def verify_and_collect(manifest):
    errors = []
    runs = manifest.get("runs") if isinstance(manifest, dict) else None
    if not isinstance(runs, list) or len(runs) != 3:
        return [], ["manifest must contain exactly three runs"]

    expected_temperatures = {0.0, 0.5, 1.0}
    actual_temperatures = {float(run.get("buyer_temperature")) for run in runs}
    if actual_temperatures != expected_temperatures:
        errors.append(
            f"manifest temperatures are {sorted(actual_temperatures)!r}, "
            f"expected {sorted(expected_temperatures)!r}"
        )

    rows = []
    negotiation_configs = []
    judge_configs = []
    for index, run in enumerate(runs, start=1):
        try:
            temperature = float(run["buyer_temperature"])
            transcript_relative = run["transcript"]
            evaluation_relative = run["evaluation"]
            transcript_path = resolved_path(ROOT, transcript_relative)
            evaluation_path = resolved_path(ROOT, evaluation_relative)
            transcript = load_json(transcript_path)
            evaluation_record = load_json(evaluation_path)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"run {index}: unable to load manifest evidence: {exc}")
            continue

        try:
            totals = transcript["totals"]
            messages = transcript["messages"]
            budget = transcript["budget"]
            meta = transcript["meta"]
            negotiation_outcome = meta["negotiation_outcome"]
            config = meta["config"]
            buyer = next(agent for agent in config["agents"] if agent["name"] == "Buyer")
            if float(buyer["temperature"]) != temperature:
                errors.append(f"run {index}: Buyer temperature disagrees with manifest")
            if len(messages) != totals["turns"]:
                errors.append(f"run {index}: message count does not match transcript totals")
            prompt_tokens = sum(message["prompt_tokens"] for message in messages)
            completion_tokens = sum(message["completion_tokens"] for message in messages)
            if prompt_tokens != totals["prompt_tokens"]:
                errors.append(f"run {index}: prompt-token sum does not match transcript totals")
            if completion_tokens != totals["completion_tokens"]:
                errors.append(f"run {index}: completion-token sum does not match transcript totals")
            if prompt_tokens + completion_tokens != budget["tokens"]:
                errors.append(f"run {index}: budget token total does not match message sums")
            negotiation_configs.append(normalized_negotiation_config(config))
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            errors.append(f"run {index}: inconsistent transcript evidence: {exc}")
            continue

        try:
            evaluation_meta = evaluation_record["meta"]
            evaluation = evaluation_record["evaluation"]
            recorded_transcript = PureWindowsPath(
                evaluation_meta["source_transcript_path"]
            )
            if recorded_transcript.name != transcript_path.name:
                errors.append(f"run {index}: evaluation references a different transcript")
            judge_configs.append(evaluation_meta["judge_config"])
            attempts = evaluation["attempts"]
            evaluation_totals = evaluation["totals"]
            input_tokens = sum(attempt["input_tokens"] for attempt in attempts)
            output_tokens = sum(attempt["output_tokens"] for attempt in attempts)
            if input_tokens != evaluation_totals["input_tokens"]:
                errors.append(f"run {index}: judge input-token sum does not match totals")
            if output_tokens != evaluation_totals["output_tokens"]:
                errors.append(f"run {index}: judge output-token sum does not match totals")
            judge_budget = evaluation["budget"]
            if input_tokens + output_tokens != judge_budget["tokens"]:
                errors.append(f"run {index}: judge budget token total does not match attempt sums")
            verdict = evaluation.get("verdict") or {}
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"run {index}: inconsistent evaluation evidence: {exc}")
            continue

        rows.append(
            {
                "temperature": temperature,
                "turns": totals["turns"],
                "negotiation_tokens": prompt_tokens + completion_tokens,
                "negotiation_seconds": budget["seconds"],
                "program_success": negotiation_outcome["success"],
                "stop_reason": budget["stop_reason"],
                "judge_score": verdict.get("score", "not accepted"),
                "judge_success": verdict.get("success", "not accepted"),
                "judge_input_tokens": input_tokens,
                "judge_output_tokens": output_tokens,
                "judge_total_tokens": input_tokens + output_tokens,
                "judge_seconds": judge_budget["seconds"],
                "transcript": transcript_relative,
                "evaluation": evaluation_relative,
                "evaluated": (
                    f"transcripts/evaluated/{Path(transcript_relative).stem}"
                    "_evaluated.json"
                ),
                "transcript_record": transcript,
                "evaluation_record": evaluation_record,
            }
        )

    if negotiation_configs and any(config != negotiation_configs[0] for config in negotiation_configs[1:]):
        errors.append("negotiation configurations differ by more than Buyer temperature")
    if judge_configs and any(config != judge_configs[0] for config in judge_configs[1:]):
        errors.append("judge configurations do not match")
    return rows, errors


def render_results(rows):
    lines = [
        "# Temperature Experiment Results",
        "",
        "This table is generated from the manifest-linked transcript and judgement JSON files. The program outcome and judge verdict are separate measurements.",
        "",
        "## Negotiation Results",
        "",
        "| Buyer temperature | Turns | Input + output tokens | Budget elapsed seconds | Program success | Stop reason | Judge score | Judge success |",
        "| ---: | ---: | ---: | ---: | :---: | :--- | ---: | :---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['temperature']:.1f} | {row['turns']} | {row['negotiation_tokens']} | "
            f"{row['negotiation_seconds']} | {str(row['program_success']).lower()} | "
            f"{row['stop_reason']} | {row['judge_score']} | {str(row['judge_success']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Judge Cost",
            "",
            "| Buyer temperature | Input tokens | Output tokens | Total tokens | Budget elapsed seconds |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['temperature']:.1f} | {row['judge_input_tokens']} | "
            f"{row['judge_output_tokens']} | {row['judge_total_tokens']} | {row['judge_seconds']} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "These are single runs at each Buyer temperature; Seller temperature remained 0.7. They do not establish a reliable temperature-effect trend. Judge scores are recorded model outputs, not verified quality ratings: manual inspection found factual errors in their explanations. Program success is reported separately.",
            "",
            "## Source Files",
            "",
            "| Buyer temperature | Transcript | Evaluation | Evaluated transcript |",
            "| ---: | :--- | :--- | :--- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['temperature']:.1f} | [transcript](../{row['transcript']}) | "
            f"[evaluation](../{row['evaluation']}) | "
            f"[evaluated transcript](../{row['evaluated']}) |"
        )
    high_temperature_row = next(row for row in rows if row["temperature"] == 1.0)
    comparison_rows = [row for row in rows if row["temperature"] != 1.0]
    comparison_tokens = " and ".join(
        str(row["negotiation_tokens"]) for row in comparison_rows
    )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "None of these three negotiations reached a valid agreement. The temperature 1.0 run used "
            f"{high_temperature_row['turns']} turns and {high_temperature_row['negotiation_tokens']} "
            f"tokens, more than the other two runs, which used {comparison_tokens} tokens. "
            "With only one run per temperature, we cannot conclude that higher temperature generally "
            "causes this. The judge explanations contained factual mistakes, so its scores cannot be "
            "treated as verified quality ratings.",
        ]
    )
    lines.extend(
        [
            "",
            "Verification: all selected files exist; transcript message counts and token sums match their saved totals; each evaluation references its selected transcript; negotiation configurations differ only in Buyer temperature; and all judge configurations match.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    with MANIFEST_PATH.open(encoding="utf-8") as manifest_file:
        manifest = yaml.safe_load(manifest_file)
    rows, errors = verify_and_collect(manifest)
    if errors:
        print("Evidence verification failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    exported_paths = export_evaluated_transcripts(rows)
    OUTPUT_PATH.write_text(render_results(rows), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    for path in exported_paths:
        print(f"Exported evaluated transcript: {path}")


if __name__ == "__main__":
    main()
