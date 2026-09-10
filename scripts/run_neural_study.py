"""Run optional neural experiments without exposing test inputs to search."""

import argparse
import json
from pathlib import Path

from therapy_switch.config import load_config
from therapy_switch.research import (
    confirm_frozen_recipe,
    prepare_synthetic_cohort,
    run_validation_search,
    select_frozen_recipe,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "search", "freeze", "confirm"])
    parser.add_argument("--root", type=Path, default=Path("artifacts/dl_search"))
    parser.add_argument("--bank", type=Path, default=Path("configs/neural_search.json"))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--confirmation-seeds", type=int, nargs="+", default=[7301, 7302, 7303])
    parser.add_argument("--names", nargs="+")
    args = parser.parse_args()
    if args.stage == "prepare":
        for seed in args.seeds:
            prepare_synthetic_cohort(load_config(args.config), args.root, seed)
    elif args.stage == "search":
        bank = json.loads(args.bank.read_text())
        if args.names:
            bank = [c for c in bank if c["name"] in args.names]
        for seed in args.seeds:
            run_validation_search(
                args.root / "datasets" / str(seed), bank, args.root / "discovery" / str(seed)
            )
    elif args.stage == "freeze":
        recipe = select_frozen_recipe(
            args.root, json.loads(args.bank.read_text()), args.seeds, args.confirmation_seeds
        )
        print(json.dumps(recipe, indent=2))
    else:
        print(json.dumps(confirm_frozen_recipe(args.root), indent=2))


if __name__ == "__main__":
    main()
