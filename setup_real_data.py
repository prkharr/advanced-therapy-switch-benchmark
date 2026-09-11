"""Create private real-data settings and SQL without overwriting existing edits."""

import argparse
from pathlib import Path

import yaml


def setup(root):
    root = Path(root).resolve()
    private = root / "configs" / "private"
    private.mkdir(parents=True, exist_ok=True)
    (root / "actual_raw_data").mkdir(parents=True, exist_ok=True)
    sources = {
        private / "benchmark.yaml": root / "configs" / "real_data.example.yaml",
        private / "delivery.yaml": root / "configs" / "delivery_real.example.yaml",
    }
    for destination, source in sources.items():
        if destination.exists():
            current = yaml.safe_load(destination.read_text(encoding="utf-8")) or {}
            updated = yaml.safe_load(source.read_text(encoding="utf-8"))
            if destination.name == "benchmark.yaml":
                for section in updated:
                    if section in current:
                        if isinstance(updated[section], dict) and isinstance(current[section], dict):
                            updated[section].update(current[section])
                        else:
                            updated[section] = current[section]
                updated["data"].update(kind="real", source="files", input_dir="../../actual_raw_data",
                                       require_export_manifest=False)
                updated["models"] = {}
            else:
                updated.update(current)
                delivery = updated.setdefault("delivery", {})
                delivery["adapter"] = "therapy_switch.delivery.adapters:load_raw"
                delivery["model"] = {
                    "trainer": "therapy_switch.models.baselines:fit_baselines",
                    "evidence_status": "independent_baseline",
                    "parameters": delivery.get("model", {}).get("parameters", {}),
                }
                delivery.pop("prepared", None)
                delivery.pop("model_artifact", None)
                updated.setdefault("extraction", {}).update(
                    sql_dir="../../sql_extraction", output_dir="../../actual_raw_data"
                )
            if updated != current:
                backup = destination.with_suffix(".yaml.bak")
                if not backup.exists():
                    backup.write_bytes(destination.read_bytes())
                destination.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")
                print(f"Updated paths and baseline settings; preserved rules: {destination.relative_to(root)}")
            else:
                print(f"Kept existing: {destination.relative_to(root)}")
        else:
            destination.write_bytes(source.read_bytes())
            print(f"Created: {destination.relative_to(root)}")
    print("Fill the null dates/rules in configs/private/benchmark.yaml and delivery.yaml.")
    print("Map the seven SQL files in sql_extraction, or download canonical CSVs to actual_raw_data.")
    print("Run: python run_pipeline.py --check")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    setup(parser.parse_args().root)
