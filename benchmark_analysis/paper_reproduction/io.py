"""Input discovery, hashing, and manifest helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

CSV_PATTERNS = ("*stress*.csv", "*benchmark*.csv", "*summary*.csv", "*grid*.csv", "*severe*.csv", "*mice*.csv", "*da*.csv", "*alignment*.csv", "*sensitivity*.csv", "*suite*.csv", "*realdata*.csv", "*mimic*.csv", "*eicu*.csv", "*.csv")


class ReproductionInputError(RuntimeError):
    """Raised when a target cannot find a required local input artifact."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("rows", "data", "results", "summary"):
                if isinstance(payload.get(key), list):
                    return pd.DataFrame(payload[key])
        raise ReproductionInputError(f"JSON file is not table-like: {path}")
    raise ReproductionInputError(f"Unsupported table format: {path}")


def _candidate_files(root: Path) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in CSV_PATTERNS:
        for path in root.rglob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                files.append(path)
    return sorted(files)


def discover_table(*, root: Path | None, explicit_input: Path | None, required_columns: Iterable[str], target_id: str, context: str) -> tuple[Path, pd.DataFrame]:
    required = set(required_columns)
    if explicit_input is not None:
        path = explicit_input.expanduser().resolve()
        if not path.exists():
            raise ReproductionInputError(f"Input for {target_id} does not exist: {path}")
        df = read_table(path)
        missing = required - set(df.columns)
        if missing:
            raise ReproductionInputError(f"Input for {target_id} is missing required columns {sorted(missing)}: {path}")
        return path, df
    if root is None:
        raise ReproductionInputError(f"{target_id} requires {context}. Pass --input or --results-root/--realdata-root.")
    root = root.expanduser().resolve()
    if not root.exists():
        raise ReproductionInputError(f"Search root for {target_id} does not exist: {root}")
    matches: list[tuple[Path, pd.DataFrame]] = []
    for path in _candidate_files(root):
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        if required.issubset(set(header.columns)):
            matches.append((path, read_table(path)))
    if not matches:
        raise ReproductionInputError(f"No input file for {target_id} under {root} has required columns {sorted(required)}. Context: {context}")
    if len(matches) > 1:
        candidates = "\n".join(f"  - {path}" for path, _ in matches[:25])
        raise ReproductionInputError(f"Multiple candidate inputs for {target_id}; pass --input to choose one:\n{candidates}")
    return matches[0]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_manifest(*, outdir: Path, target_id: str, latex_label: str, caption: str, argv: list[str], inputs: list[Path], outputs: list[Path], source_type: str, notes: list[str] | None = None) -> Path:
    manifest = {
        "target_id": target_id,
        "latex_label": latex_label,
        "caption": caption,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": argv,
        "inputs": [{"path": str(path), "sha256": sha256_file(path) if path.exists() and path.is_file() else None} for path in inputs],
        "outputs": [str(path) for path in outputs],
        "source_type": source_type,
        "notes": notes or [],
    }
    path = outdir / f"{target_id}_manifest.json"
    write_json(path, manifest)
    return path
