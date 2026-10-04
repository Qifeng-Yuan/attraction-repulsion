"""Build forecast panels from original keyword counts and project support."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import pickle
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "runs" / "prepared" / "panel_nsfc.pkl.gz"
DEFAULT_OUTPUT = ROOT / "runs" / "panel.pkl.gz"
ORIGINS = range(1990, 2025)
HORIZONS = (1, 5, 10)
MIN_SUPPORT = 30
MAX_HISTORY = 12
VOCAB_MAX = 12_000
RECENT_MAX = 150


def load_panel(path: str | Path = DEFAULT_OUTPUT) -> dict:
    """Load a compressed forecast panel."""
    with gzip.open(Path(path), "rb") as handle:
        return pickle.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _dump_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _sorted_terms(counter: Counter, minimum: int, limit: int) -> list[str]:
    values = [(term, int(n)) for term, n in counter.items() if n >= minimum]
    values.sort(key=lambda item: (-item[1], item[0]))
    return [term for term, _ in values[:limit]]


def _normalise_rows(weighted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(weighted, axis=1)
    z = np.divide(
        weighted,
        norms[:, None],
        out=np.zeros_like(weighted, dtype=np.float32),
        where=norms[:, None] > 1e-12,
    )
    return z.astype(np.float32, copy=False), norms


def _similarity(prevalence: np.ndarray, idf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, norms = _normalise_rows(prevalence * idf[None, :])
    result = np.clip(z @ z.T, 0.0, 1.0).astype(np.float32)
    return result, z, norms


def _pairs(matrix: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    return matrix[pair_i, pair_j].astype(np.float32, copy=False)


class PanelBuilder:
    def __init__(self, raw: dict):
        self.raw = raw
        self.years = np.asarray(raw["years"], dtype=np.int16)
        self.year_to_i = {int(year): i for i, year in enumerate(self.years)}
        self.codes = list(raw["codes"])
        self.counts = raw["nsfc_counts"]
        self.n = np.asarray(raw["nsfc_n"], dtype=np.float64)
        self.c = len(self.codes)
        self.pair_i, self.pair_j = np.triu_indices(self.c, 1)

        self.global_by_year: list[Counter] = []
        self.cumulative: list[Counter] = []
        running = Counter()
        for rows in self.counts:
            current = Counter()
            for row in rows:
                current.update(row)
            self.global_by_year.append(current)
            running = running + current
            self.cumulative.append(running.copy())

    def vocabulary(self, ti: int) -> tuple[list[str], np.ndarray]:
        cumulative = self.cumulative[ti]
        recent = Counter()
        for ai in range(max(0, ti - 2), ti + 1):
            recent.update(self.global_by_year[ai])
        main = _sorted_terms(cumulative, 5, VOCAB_MAX)
        extra = _sorted_terms(recent, 3, RECENT_MAX)
        terms = list(dict.fromkeys(main + extra))
        term_to_j = {term: j for j, term in enumerate(terms)}
        presence = np.zeros((self.c, len(terms)), dtype=bool)
        for ai in range(ti + 1):
            for ci, row in enumerate(self.counts[ai]):
                for term in row:
                    j = term_to_j.get(term)
                    if j is not None:
                        presence[ci, j] = True
        historical_codes = int(np.count_nonzero(self.n[: ti + 1].sum(axis=0) > 0))
        df = presence.sum(axis=0)
        idf = (np.log((historical_codes + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
        return terms, idf

    def annual_matrices(self, term_to_j: dict[str, int]) -> list[np.ndarray]:
        matrices: list[np.ndarray] = []
        k = len(term_to_j)
        for ai in range(len(self.years)):
            matrix = np.zeros((self.c, k), dtype=np.float32)
            for ci, row in enumerate(self.counts[ai]):
                for term, value in row.items():
                    j = term_to_j.get(term)
                    if j is not None:
                        matrix[ci, j] = value
            matrices.append(matrix)
        return matrices

    def pooled(
        self, annual_counts: list[np.ndarray], start_i: int, end_i: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = np.add.reduce(annual_counts[start_i : end_i + 1])
        n = self.n[start_i : end_i + 1].sum(axis=0)
        p = np.divide(
            count,
            n[:, None],
            out=np.zeros_like(count, dtype=np.float32),
            where=n[:, None] > 0,
        )
        return p.astype(np.float32, copy=False), count.astype(np.float32, copy=False), n.astype(np.float32)

    def predicted_project_n(self, ti: int, q: int) -> np.ndarray:
        if q <= 0:
            return self.n[ti + q].astype(np.float32)
        start = max(0, ti - 4)
        hist = self.n[start : ti + 1]
        x = np.arange(len(hist), dtype=np.float64)
        if len(hist) < 2:
            return hist[-1].astype(np.float32)
        xc = x - x.mean()
        slope = (xc[:, None] * hist).sum(axis=0) / max(float((xc * xc).sum()), 1e-12)
        estimate = hist[-1] + q * slope
        return np.maximum(estimate, 0.0).astype(np.float32)

    def count_forecasts(
        self,
        ti: int,
        h: int,
        annual_counts: list[np.ndarray],
        base_r: np.ndarray,
        idf: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        _, base_count, base_n = self.pooled(annual_counts, ti - 2, ti)
        base_share = np.divide(base_count, base_n[:, None], out=np.zeros_like(base_count), where=base_n[:, None] > 0)
        last_share = np.divide(
            annual_counts[ti], self.n[ti, :, None], out=np.zeros_like(annual_counts[ti]), where=self.n[ti, :, None] > 0
        )
        hist_start = max(0, ti - 4)
        annual_share = []
        for ai in range(hist_start, ti + 1):
            annual_share.append(
                np.divide(
                    annual_counts[ai], self.n[ai, :, None], out=np.zeros_like(annual_counts[ai]), where=self.n[ai, :, None] > 0
                )
            )
        shares = np.stack(annual_share)
        x = np.arange(len(shares), dtype=np.float32)
        xc = x - x.mean()
        slope = (xc[:, None, None] * shares).sum(axis=0) / max(float((xc * xc).sum()), 1e-12)

        values = []
        known_n = np.zeros(self.c, dtype=np.float32)
        total_n = np.zeros(self.c, dtype=np.float32)
        for kind, multiplier in (("base", 0.0), ("last", 0.0), ("linear", 0.25), ("linear", 0.5), ("linear", 1.0)):
            summed_count = np.zeros_like(base_count)
            summed_n = np.zeros(self.c, dtype=np.float32)
            for q in range(h - 2, h + 1):
                if q <= 0:
                    nq = self.n[ti + q].astype(np.float32)
                    cq = annual_counts[ti + q]
                else:
                    nq = self.predicted_project_n(ti, q)
                    if kind == "base":
                        share = base_share
                    elif kind == "last":
                        share = last_share
                    else:
                        share = np.clip(last_share + multiplier * q * slope, 0.0, 1.0)
                    cq = share * nq[:, None]
                summed_count += cq
                summed_n += nq
            p_hat = np.divide(
                summed_count, summed_n[:, None], out=np.zeros_like(summed_count), where=summed_n[:, None] > 0
            )
            r_hat, _, _ = _similarity(p_hat, idf)
            values.append(_pairs(r_hat - base_r, self.pair_i, self.pair_j))

        for q in range(h - 2, h + 1):
            nq = self.n[ti + q].astype(np.float32) if q <= 0 else self.predicted_project_n(ti, q)
            total_n += nq
            if q <= 0:
                known_n += nq
        known_fraction = np.divide(known_n, total_n, out=np.zeros_like(known_n), where=total_n > 0)
        stacked = np.stack(values, axis=1).astype(np.float32)
        if h - 2 > 0:
            # Constant shares preserve cosine similarity; remove round-off noise.
            stacked[:, 0] = 0.0
        return stacked, known_fraction.astype(np.float32)

    def origin(self, origin: int) -> tuple[dict, dict]:
        ti = self.year_to_i[origin]
        terms, idf = self.vocabulary(ti)
        term_to_j = {term: j for j, term in enumerate(terms)}
        annual_counts = self.annual_matrices(term_to_j)

        annual_years = self.years[max(0, ti - MAX_HISTORY + 1) : ti + 1].copy()
        annual_r = []
        for year in annual_years:
            ai = self.year_to_i[int(year)]
            p, _, _ = self.pooled(annual_counts, ai, ai)
            annual_r.append(_similarity(p, idf)[0])

        rolling_first = max(2, ti - MAX_HISTORY + 1)
        rolling_years = self.years[rolling_first : ti + 1].copy()
        rolling_r = []
        rolling_valid = []
        for year in rolling_years:
            ai = self.year_to_i[int(year)]
            p, _, _ = self.pooled(annual_counts, ai - 2, ai)
            r_value, _, r_norm = _similarity(p, idf)
            rolling_r.append(r_value)
            rolling_valid.append(r_norm > 1e-12)

        p, base_count, n3 = self.pooled(annual_counts, ti - 2, ti)
        base_r, u, base_norm = _similarity(p, idf)
        z = (p * idf[None, :]).astype(np.float32)
        eligible_codes = (n3 >= MIN_SUPPORT) & (base_norm > 1e-12)
        eligible = eligible_codes[self.pair_i] & eligible_codes[self.pair_j]

        prior_start = max(0, ti - 5)
        prior_end = max(0, ti - 3)
        prior_p, _, _ = self.pooled(annual_counts, prior_start, prior_end)
        last_p, _, _ = self.pooled(annual_counts, ti, ti)

        labels = {}
        future = {}
        counters = {}
        known_fraction = {}
        label_audit = {}
        for h in HORIZONS:
            counter_values, fraction = self.count_forecasts(ti, h, annual_counts, base_r, idf)
            counters[h] = {
                "values": counter_values,
                "names": ["base_share", "last_share", "linear_share_x0.25", "linear_share_x0.5", "linear_share_x1"],
            }
            known_fraction[h] = fraction
            if origin + h <= int(self.years[-1]):
                target_i = self.year_to_i[origin + h]
                target_p, target_count, target_n = self.pooled(annual_counts, target_i - 2, target_i)
                target_r, _, target_norm = _similarity(target_p, idf)
                target_good = (target_n >= MIN_SUPPORT) & (target_norm > 1e-12)
                mask = eligible & target_good[self.pair_i] & target_good[self.pair_j]
                labels[h] = {
                    "delta": _pairs(target_r - base_r, self.pair_i, self.pair_j),
                    "mask": mask.astype(bool),
                    "target_R": _pairs(target_r, self.pair_i, self.pair_j),
                }
                future[h] = {"p": target_p, "count": target_count, "n": target_n}
                label_audit[str(h)] = {
                    "target_end_year": origin + h,
                    "mask_pairs": int(mask.sum()),
                    "target_supported_codes": int(target_good.sum()),
                }

        item = {
            "annual_years": annual_years,
            "annual_R": np.stack(annual_r).astype(np.float32),
            "rolling_years": rolling_years,
            "rolling_R": np.stack(rolling_r).astype(np.float32),
            "rolling_valid": np.stack(rolling_valid).astype(bool),
            "z": z,
            "u": u,
            "idf": idf,
            "n3": n3,
            "eligible": eligible.astype(bool),
            "base_R": _pairs(base_r, self.pair_i, self.pair_j),
            "labels": labels,
            "counter": counters,
            "known_fraction": known_fraction,
            "topic": {
                "terms": terms,
                "p": p,
                "last_p": last_p,
                "prior_p": prior_p,
                "future": future,
            },
        }
        audit = {
            "origin": origin,
            "vocabulary_size": len(terms),
            "annual_years": [int(x) for x in annual_years],
            "rolling_years": [int(x) for x in rolling_years],
            "history_shortfall_annual": max(0, MAX_HISTORY - len(annual_years)),
            "history_shortfall_rolling": max(0, MAX_HISTORY - len(rolling_years)),
            "eligible_codes": int(eligible_codes.sum()),
            "eligible_pairs": int(eligible.sum()),
            "labels": label_audit,
            "known_fraction_range": {
                str(h): [float(known_fraction[h].min()), float(known_fraction[h].max())] for h in HORIZONS
            },
        }
        return item, audit


def build(input_path: Path, output_path: Path) -> dict:
    started = time.time()
    with gzip.open(input_path, "rb") as handle:
        raw = pickle.load(handle)
    builder = PanelBuilder(raw)
    result = {
        "codes": builder.codes,
        "pair_i": builder.pair_i.astype(np.int16),
        "pair_j": builder.pair_j.astype(np.int16),
        "origins": {},
    }
    origin_audit = []
    for origin in ORIGINS:
        print(f"panel origin={origin}", flush=True)
        result["origins"][origin], audit = builder.origin(origin)
        origin_audit.append(audit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.name + ".tmp")
    with gzip.open(temporary_output, "wb", compresslevel=5) as handle:
        pickle.dump(result, handle, pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_output, output_path)

    support_path = output_path.parent / "annual_support_audit.csv"
    with support_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["year", "code", "nsfc_n", "term_document_frequency_sum", "distinct_terms"])
        writer.writeheader()
        for ai, year in enumerate(builder.years):
            for ci, code in enumerate(builder.codes):
                row = builder.counts[ai][ci]
                writer.writerow(
                    {
                        "year": int(year),
                        "code": code,
                        "nsfc_n": int(builder.n[ai, ci]),
                        "term_document_frequency_sum": int(sum(row.values())),
                        "distinct_terms": len(row),
                    }
                )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(input_path.resolve()),
        "source_sha256": _sha256(input_path),
        "output": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "loader": "build_panel.load_panel",
        "shape": {"codes": builder.c, "pairs": len(builder.pair_i), "origins": len(result["origins"])},
        "years": [int(builder.years[0]), int(builder.years[-1])],
        "origins": [min(ORIGINS), max(ORIGINS)],
        "horizons": list(HORIZONS),
        "source_fields": ["nsfc_counts", "nsfc_n"],
        "count_semantics": "nsfc_n is projects with at least one original keyword; counts are original-keyword document frequencies",
        "vocabulary": "at each origin: cumulative original-keyword DF>=5 top12000 plus trailing-3-year DF>=3 top150",
        "idf": "at each origin only: log((number of historically observed codes+1)/(codes containing term through origin+1))+1",
        "target": "cosine(target 3-year original-keyword prevalence, origin vocabulary/IDF) minus origin 3-year cosine",
        "eligibility": "both base and target code windows have nsfc_n>=30 and nonzero weighted vectors; future support only enters mature labels",
        "history": "annual and rolling arrays end at origin, contain at most 12 actual snapshots, and are never backfilled from future years",
        "counter": "five count-aware deltas; target window years q=h-2..h, q<=0 observed and q>0 projected; future project N is an OLS linear extrapolation from the latest up-to-5 annual nsfc_n values, clipped at zero; future term shares are base-3-year share, last-year share, or last-year share plus 0.25/0.5/1 times q times the latest up-to-5-year annual-share OLS slope, clipped to [0,1]; no mature future counts enter predictors",
        "threads_max": 2,
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "seconds": time.time() - started,
        "origin_audit": origin_audit,
        "checks": {
            "origin_1990_h10_has_label": 10 in result["origins"][1990]["labels"],
            "origin_1990_h10_mask_pairs": int(result["origins"][1990]["labels"][10]["mask"].sum()),
            "origin_2014_h10_has_label": 10 in result["origins"][2014]["labels"],
            "origin_2024_has_no_labels": len(result["origins"][2024]["labels"]) == 0,
            "origin_2024_h5_known_fraction_zero": bool(np.all(result["origins"][2024]["known_fraction"][5] == 0)),
            "origin_2024_h10_known_fraction_zero": bool(np.all(result["origins"][2024]["known_fraction"][10] == 0)),
            "all_unknown_base_share_delta_exact_zero": all(
                np.all(item["counter"][h]["values"][:, 0] == 0)
                for item in result["origins"].values()
                for h in (5, 10)
            ),
            "all_counter_finite": all(
                np.isfinite(item["counter"][h]["values"]).all()
                for item in result["origins"].values()
                for h in HORIZONS
            ),
            "z_is_raw_weighted_state": all(
                np.allclose(item["z"], item["topic"]["p"] * item["idf"][None, :])
                for item in result["origins"].values()
            ),
        },
    }
    manifest["training_evaluable_pairs"] = {
        str(t): {
            str(h): int(
                sum(
                    result["origins"][v]["labels"][h]["mask"].sum()
                    for v in range(1990, t - h + 1)
                    if h in result["origins"][v]["labels"]
                )
            )
            for h in HORIZONS
        }
        for t in range(1990, 2025)
    }
    _dump_json(output_path.parent / "panel_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        manifest = build(args.input, args.output)
    print(json.dumps({"output": manifest["output"], "sha256": manifest["output_sha256"], "checks": manifest["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
