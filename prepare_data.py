"""Prepare original NSFC keyword counts for 1987--2024."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
YEARS = np.arange(1987, 2025, dtype=np.int16)
YEAR_TO_I = {int(year): i for i, year in enumerate(YEARS)}
CODE_RE = re.compile(r"(?<![A-Z0-9])F\d{4}(?!\d)", re.IGNORECASE)
FILE_YEAR_RE = re.compile(r"_(19\d{2}|20\d{2})\.jsonl$")


def normalise_term(value) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).lower().replace("***", "").strip()


def original_keywords(value) -> set[str]:
    if isinstance(value, str):
        values = re.split(r"[;；]", value)
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = ()
    return {term for raw in values if (term := normalise_term(raw))}


def prepare(input_dir: Path, output: Path) -> dict:
    paths = sorted(input_dir.glob("nsfc_final_projects_F_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No nsfc_final_projects_F_*.jsonl files in {input_dir}")

    counts = defaultdict(Counter)
    support = Counter()
    seen_ids = set()
    codes = set()
    stats = Counter()
    for path in paths:
        match = FILE_YEAR_RE.search(path.name)
        filename_year = int(match.group(1)) if match else None
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                stats["rows_read"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["malformed_json"] += 1
                    continue
                if not isinstance(record, dict):
                    raise ValueError(f"Each JSONL record must be an object: {path.name}")
                try:
                    year = int(record.get("conclusion_year"))
                except (TypeError, ValueError):
                    year = filename_year
                    stats["year_fallback_to_filename"] += 1
                if year not in YEAR_TO_I:
                    stats["outside_year_range"] += 1
                    continue
                project_id = str(record.get("project_id") or "").strip()
                if project_id:
                    if project_id in seen_ids:
                        stats["duplicate_id"] += 1
                        continue
                    seen_ids.add(project_id)
                else:
                    stats["missing_id_included"] += 1
                code_text = unicodedata.normalize(
                    "NFKC", str(record.get("application_code") or "")
                ).upper()
                code_match = CODE_RE.search(code_text)
                if code_match is None:
                    stats["invalid_or_missing_code"] += 1
                    continue
                code = code_match.group(0)
                codes.add(code)
                stats["included_projects"] += 1
                terms = original_keywords(record.get("keywords"))
                if terms:
                    support[year, code] += 1
                    counts[year, code].update(terms)
                    stats["projects_with_original_keywords"] += 1
                else:
                    stats["missing_original_keywords"] += 1

    codes = sorted(codes)
    if len(codes) < 2:
        raise ValueError("At least two valid F-codes are required")
    raw = {
        "years": YEARS.copy(),
        "codes": codes,
        "nsfc_counts": [[counts[int(year), code] for code in codes] for year in YEARS],
        "nsfc_n": np.asarray(
            [[support[int(year), code] for code in codes] for year in YEARS], dtype=np.int64
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=5) as handle:
        pickle.dump(raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, output)
    manifest = {
        "years": [1987, 2024],
        "source_files": [path.name for path in paths],
        "stats": dict(stats),
        "codes": len(codes),
        "keyword_document_frequency_sum": sum(
            sum(counter.values()) for row in raw["nsfc_counts"] for counter in row
        ),
        "fields_used": ["project_id", "conclusion_year", "application_code", "keywords"],
        "support": "projects with at least one original keyword",
        "count": "one occurrence per keyword per project after normalization",
    }
    output.with_name("preparation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / "prepared" / "panel_nsfc.pkl.gz")
    args = parser.parse_args()
    manifest = prepare(args.input_dir.resolve(), args.output.resolve())
    print(json.dumps({"output": str(args.output), "stats": manifest["stats"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
