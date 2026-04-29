#!/usr/bin/env python3
"""Export the canonical SynLV scenario registry as Markdown and JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_analysis.synlv_release_config import markdown_table, registry_as_dict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SynLV scenario registry docs.")
    parser.add_argument("--markdown", type=Path, default=REPO_ROOT / "docs" / "SCENARIO_REGISTRY.md")
    parser.add_argument("--json", dest="json_path", type=Path, default=REPO_ROOT / "docs" / "scenario_registry.json")
    args = parser.parse_args()

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_table() + "\n", encoding="utf-8")
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(registry_as_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.markdown}")
    print(f"wrote {args.json_path}")


if __name__ == "__main__":
    main()
