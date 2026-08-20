"""Command-line interface for generation, validation, and benchmarking."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Sequence

from therapy_switch import __version__
from therapy_switch.config import load_config, validate_config
from therapy_switch.data import (
    build_cohort,
    build_event_sequences,
    generate_synthetic_claims,
    validate_cohort_timeline,
)
from therapy_switch.features import build_tabular_features
from therapy_switch.io import load_claims_directory, save_claims_directory
from therapy_switch.pipeline import run_pipeline
from therapy_switch.schemas import validate_tables


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="therapy-switch",
        description="Classical ML vs longitudinal DL claims benchmark.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run configured benchmark experiments.")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--artifact-dir", type=Path)
    run.add_argument(
        "--experiment",
        choices=("stratified", "temporal", "both"),
        default=None,
        help="Override configured split experiments for this run.",
    )
    run.add_argument("--no-plots", action="store_true")

    generate = subparsers.add_parser(
        "generate", help="Generate canonical, entirely synthetic claims tables."
    )
    generate.add_argument("--config", required=True, type=Path)
    generate.add_argument("--output-dir", required=True, type=Path)
    generate.add_argument("--format", choices=("csv", "parquet"), default="csv")

    validate = subparsers.add_parser(
        "validate-data", help="Validate schema, timeline, features, and sequences without training."
    )
    validate.add_argument("--config", required=True, type=Path)
    return parser


def _run_command(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    config = copy.deepcopy(config)
    if arguments.output_dir is not None:
        config["project"]["output_dir"] = str(arguments.output_dir)
    if arguments.artifact_dir is not None:
        config["project"]["artifact_dir"] = str(arguments.artifact_dir)
    if arguments.experiment:
        experiments = (
            ["stratified", "temporal"] if arguments.experiment == "both" else [arguments.experiment]
        )
        config["splitting"]["experiments"] = experiments
        if config["splitting"].get("primary_experiment") not in experiments:
            config["splitting"]["primary_experiment"] = experiments[-1]
    if arguments.no_plots:
        config.setdefault("visualizations", {})["enabled"] = False
    validate_config(config)
    result = run_pipeline(config)
    primary = result.experiments[result.primary_experiment]
    payload: dict[str, Any] = {
        "status": "COMPLETED",
        "primary_experiment": result.primary_experiment,
        "eligible_patients": len(result.cohort),
        "prevalence": float(result.cohort["label"].mean()),
        "recommended_model": primary.recommendation.model,
        "decision": primary.recommendation.decision,
        "output_dir": str(result.output_dir.resolve()),
        "artifact_dir": str(result.artifact_dir.resolve()),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _generate_command(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    tables = generate_synthetic_claims(config)
    save_claims_directory(tables, arguments.output_dir, file_format=arguments.format)
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "output_dir": str(arguments.output_dir.resolve()),
                "rows": {name: len(frame) for name, frame in tables.items()},
                "notice": "Entirely synthetic; not derived from proprietary claims.",
            },
            indent=2,
        )
    )
    return 0


def _validate_command(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    if str(config["data"].get("source", "synthetic")).lower() == "synthetic":
        tables = generate_synthetic_claims(config)
    else:
        tables = load_claims_directory(config)
    validate_tables(tables)
    cohort = build_cohort(tables, config)
    validate_cohort_timeline(cohort, tables, config)
    features = build_tabular_features(tables, cohort, config)
    sequences = build_event_sequences(tables, cohort, config)
    sequences.to_sequence_split().validated(expected_rows=len(cohort))
    print(
        json.dumps(
            {
                "status": "VALID",
                "eligible_patients": len(cohort),
                "positive_patients": int(cohort["label"].sum()),
                "prevalence": float(cohort["label"].mean()),
                "feature_count": len(features.columns) - 3,
                "sequence_patients": len(sequences),
                "sequence_max_length": int(sequences.attention_mask.shape[1]),
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        return _run_command(arguments)
    if arguments.command == "generate":
        return _generate_command(arguments)
    if arguments.command == "validate-data":
        return _validate_command(arguments)
    raise AssertionError(f"Unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
