"""Create private real-data settings and SQL without overwriting existing edits."""

import argparse
from pathlib import Path


def setup(root):
    root = Path(root).resolve()
    private = root / "configs" / "private"
    (private / "sql").mkdir(parents=True, exist_ok=True)
    sources = {
        private / "benchmark.yaml": root / "configs" / "real_data.example.yaml",
        private / "delivery.yaml": root / "configs" / "delivery_real.example.yaml",
    }
    sources.update({private / "sql" / p.name: p for p in (root / "sql/raw_extract").glob("*.sql")})
    for destination, source in sources.items():
        if destination.exists():
            print(f"Kept existing: {destination.relative_to(root)}")
        else:
            destination.write_bytes(source.read_bytes())
            print(f"Created: {destination.relative_to(root)}")
    print("Fill the null dates/rules in configs/private/benchmark.yaml and delivery.yaml.")
    print("Map all seven configs/private/sql/*.sql queries to reviewed source data.")
    print("Run: python run_pipeline.py --check --extract")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    setup(parser.parse_args().root)
