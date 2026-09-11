"""Reproducible code and package evidence for each real-data run."""

import hashlib
from importlib.metadata import version
from pathlib import Path


def source_fingerprint():
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def runtime_versions():
    return {name: version(name) for name in ("pandas", "numpy", "scikit-learn", "lightgbm", "joblib")}
