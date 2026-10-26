"""Evaluate a saved negotiation transcript with the judge.

Prepared with AI assistance.
"""

import argparse
import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from judge import judge
from llm_client import make_client


def save_evaluation(
    transcript_path, config_path, record, config, evaluation, evaluation_started_at
):
    output_dir = Path("transcripts/judgements")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = Path(transcript_path).stem
    while True:
        output_path = output_dir / f"{output_stem}_judgement_{uuid.uuid4().hex}.json"
        try:
            with output_path.open("x", encoding="utf-8") as output_file:
                json.dump(
                    {
                        "meta": {
                            "source_transcript_path": str(
                                Path(transcript_path).resolve()
                            ),
                            "source_mock_mode": record.get("meta", {}).get(
                                "mock_mode"
                            ),
                            "judge_config_path": str(Path(config_path).resolve()),
                            "judge_config": copy.deepcopy(config),
                            "evaluation_started_at": evaluation_started_at,
                        },
                        "evaluation": evaluation,
                    },
                    output_file,
                    indent=2,
                )
            return output_path
        except FileExistsError:
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved negotiation transcript."
    )
    parser.add_argument(
        "--transcript", required=True, help="path to the saved transcript JSON"
    )
    parser.add_argument(
        "--config",
        default="configs/judge.yaml",
        help="path to the judge YAML configuration",
    )
    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    config_path = Path(args.config)
    with transcript_path.open(encoding="utf-8") as transcript_file:
        record = json.load(transcript_file)
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    client = make_client(
        mock=False,
        options=config.get("model_options"),
        response_format=config.get("response_format"),
    )
    evaluation_started_at = datetime.now(timezone.utc).isoformat()
    evaluation = judge(record, client, config)
    output_path = save_evaluation(
        transcript_path,
        config_path,
        record,
        config,
        evaluation,
        evaluation_started_at,
    )

    print(f"Status: {evaluation['status']}")
    if evaluation.get("verdict") is not None:
        print(f"Verdict: {json.dumps(evaluation['verdict'])}")
    totals = evaluation["totals"]
    print(
        "Judge tokens: "
        f"{totals['input_tokens']} input, {totals['output_tokens']} output"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()