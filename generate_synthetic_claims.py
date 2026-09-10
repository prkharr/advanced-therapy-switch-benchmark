"""Generate the demo's seven synthetic raw CSV tables without training a model.

Run from the cloned repository after installing it with: python -m pip install -e .
Example: python generate_synthetic_claims.py --patients 1200 --seed 42
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from therapy_switch.data.generate_synthetic_claims import generate_synthetic_claims
from therapy_switch.delivery.pipeline import load_delivery_config


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/delivery_demo.yaml")
    parser.add_argument("--patients", type=int, help="Override the configured patient count")
    parser.add_argument("--seed", type=int, help="Override the configured random seed")
    parser.add_argument("--output-dir", type=Path, default=root / "data/synthetic/raw")
    args = parser.parse_args(argv)

    if args.patients is not None and args.patients < 2:
        parser.error("--patients must be at least 2")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")

    config = load_delivery_config(args.config)
    if config["data"]["source"] != "synthetic":
        parser.error("Use a delivery configuration with data.source set to synthetic")
    if args.patients is not None:
        config["data"].setdefault("synthetic", {})["n_patients"] = args.patients
    if args.seed is not None:
        config["project"]["random_seed"] = args.seed

    print("Generating synthetic claims...", flush=True)
    tables = generate_synthetic_claims(config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    destination = args.output_dir.resolve() / run_id
    destination.mkdir(parents=True, exist_ok=False)
    row_counts = {}
    for name, frame in tables.items():
        frame.to_csv(destination / f"{name}.csv", index=False)
        row_counts[name] = len(frame)
        print(f"  {name}.csv: {len(frame):,} rows", flush=True)

    manifest = {
        "status": "COMPLETED",
        "data_scope": "Synthetic demonstration",
        "run_id": run_id,
        "seed": config["project"]["random_seed"],
        "row_counts": row_counts,
        "raw_csv_dir": str(destination),
    }
    (destination / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Completed. CSV files saved to: {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
