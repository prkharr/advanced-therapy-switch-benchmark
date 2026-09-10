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
    generate_synthetic_claims,
)
from therapy_switch.io import save_claims_directory
from therapy_switch.pipeline import prepare_inputs, run_pipeline


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
    run.add_argument("--raw-dir", type=Path, help="Use a completed canonical CSV extract")
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
    validate.add_argument("--raw-dir", type=Path, help="Use a completed canonical CSV extract")
    return parser


def _run_command(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    config = copy.deepcopy(config)
    if arguments.raw_dir:
        config["data"].update(source="files", input_dir=str(arguments.raw_dir.resolve()), file_format="csv")
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
        "eligible_patients": int(result.cohort.patient_id.nunique()),
        "eligible_snapshots": len(result.cohort),
        "model_statuses": dict(zip(primary.benchmark.Model, primary.benchmark.Status)),
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
    if config["data"].get("kind") == "real" or config["data"]["source"] != "synthetic":
        raise ValueError("Synthetic generation requires an explicit synthetic configuration")
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
    if arguments.raw_dir:
        config["data"].update(source="files", input_dir=str(arguments.raw_dir.resolve()), file_format="csv")
    _, inputs = prepare_inputs(config)
    print(
        json.dumps(
            {
                "status": "VALID",
                "eligible_patients": int(inputs.snapshots.patient_id.nunique()),
                "eligible_snapshots": len(inputs.snapshots),
                "positive_snapshots": int(inputs.snapshots.label.sum()),
                "prevalence": float(inputs.snapshots.label.mean()),
                "feature_count": len(inputs.feature_columns),
                "event_rows": len(inputs.events),
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
