"""Independent emerging code-keyword forecasts with temporal model selection."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import pickle
import re
import time
from pathlib import Path
from datetime import datetime, timezone

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "emergence"
PANEL_PATH = ROOT / "runs" / "panel.pkl.gz"
PANEL_MANIFEST = ROOT / "runs" / "panel_manifest.json"
DESIGN_PATH = OUT / "design_manifest.json"
CODE_NAMES_FILE = None
SEED = 20260909
HORIZONS = (1, 5, 10)
FEATURE_NAMES = [
    "base_share",
    "last_year_share",
    "prior_three_year_share",
    "recent_share_delta",
    "last_vs_prior_delta",
    "log_growth_ratio",
    "log_base_keyword_count",
    "log_code_n3",
    "within_code_relative_share",
    "log_global_keyword_count",
    "code_breadth_fraction",
    "growing_code_fraction",
    "other_code_mean_delta",
    "global_max_recent_delta",
]
THRESHOLDS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5)
OUTER = {
    1: list(range(2018, 2024)),
    5: list(range(2014, 2020)),
    10: list(range(2010, 2015)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_inputs():
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    panel_manifest = json.loads(PANEL_MANIFEST.read_text(encoding="utf-8"))
    actual_hash = sha256(PANEL_PATH)
    expected = panel_manifest["output_sha256"]
    if actual_hash != expected:
        raise RuntimeError(f"panel hash mismatch: {actual_hash} != {expected}")
    if actual_hash != design["panel"]["expected_sha256"]:
        design["panel"]["resolved_sha256"] = actual_hash
        design["panel"]["hash_changed_from_initial_freeze"] = True
    with gzip.open(PANEL_PATH, "rb") as handle:
        panel = pickle.load(handle)
    return design, panel_manifest, panel, actual_hash


def universe_and_labels(item: dict, horizon: int | None):
    topic = item["topic"]
    p = np.asarray(topic["p"], dtype=np.float32)
    c, k = p.shape
    nb = np.rint(np.asarray(item["n3"])).astype(np.int64)
    cb = np.rint(p * nb[:, None]).astype(np.int64)
    base_ok = nb >= 30
    mask = base_ok[:, None] & (20 * cb <= nb[:, None])
    y = None
    if horizon is not None:
        future = topic["future"][horizon]
        nt = np.rint(np.asarray(future["n"])).astype(np.int64)
        ct = np.rint(np.asarray(future["count"])).astype(np.int64)
        target_ok = nt >= 30
        mask &= target_ok[:, None]
        y = (
            (100 * (ct * nb[:, None] - cb * nt[:, None]) >= nb[:, None] * nt[:, None])
            & (2 * ct * nb[:, None] >= 3 * cb * nt[:, None])
            & (ct >= 3)
        )
    idx = np.flatnonzero(mask.ravel())
    return idx, None if y is None else y.ravel()[idx].astype(np.uint8)


def audit_float_boundaries(panel: dict) -> dict:
    affected = []
    total_mask = 0
    total_event = 0
    for origin, item in panel["origins"].items():
        p = np.asarray(item["topic"]["p"], dtype=np.float32)
        nb = np.rint(np.asarray(item["n3"])).astype(np.int64)
        cb = np.rint(p * nb[:, None]).astype(np.int64)
        for horizon, future in item["topic"]["future"].items():
            fp = np.asarray(future["p"], dtype=np.float32)
            fc = np.asarray(future["count"], dtype=np.float32)
            nt = np.rint(np.asarray(future["n"])).astype(np.int64)
            ct = np.rint(fc).astype(np.int64)
            float_mask = (nb[:, None] >= 30) & (nt[:, None] >= 30) & (p <= 0.05)
            exact_mask = (nb[:, None] >= 30) & (nt[:, None] >= 30) & (20 * cb <= nb[:, None])
            float_event = float_mask & (fp - p >= 0.01) & (fp >= 1.5 * p) & (fc >= 3)
            exact_event = exact_mask & (
                100 * (ct * nb[:, None] - cb * nt[:, None]) >= nb[:, None] * nt[:, None]
            ) & (2 * ct * nb[:, None] >= 3 * cb * nt[:, None]) & (ct >= 3)
            mask_diff = int(np.sum(float_mask != exact_mask))
            event_diff = int(np.sum(float_event != exact_event))
            total_mask += mask_diff
            total_event += event_diff
            if mask_diff or event_diff:
                affected.append(
                    {
                        "origin": int(origin),
                        "horizon": int(horizon),
                        "base_mask_differences": mask_diff,
                        "event_label_differences": event_diff,
                    }
                )
    return {
        "integer_rule_is_current": True,
        "float32_rule_is_archived": True,
        "total_base_mask_differences": total_mask,
        "total_event_label_differences": total_event,
        "affected": affected,
        "exact_rules": {
            "base_share": "20*Cb <= Nb",
            "absolute_increase": "100*(Ct*Nb-Cb*Nt) >= Nb*Nt",
            "ratio": "2*Ct*Nb >= 3*Cb*Nt",
            "count": "Ct >= 3",
            "integer_type": "int64",
        },
    }


def feature_matrix(item: dict, indices: np.ndarray, horizon_feature: int | None = None) -> np.ndarray:
    topic = item["topic"]
    p = np.asarray(topic["p"], dtype=np.float32)
    last = np.asarray(topic["last_p"], dtype=np.float32)
    prior = np.asarray(topic["prior_p"], dtype=np.float32)
    n3 = np.asarray(item["n3"], dtype=np.float32)
    c, k = p.shape
    ci = indices // k
    ki = indices % k
    delta = p - prior
    base_count = p * n3[:, None]
    max_within = np.maximum(p.max(axis=1), 1e-8)
    global_count = base_count.sum(axis=0)
    breadth = (p > 0).mean(axis=0)
    growing = (delta >= 0.01).mean(axis=0)
    other_mean = (delta.sum(axis=0)[None, :] - delta) / max(c - 1, 1)
    max_delta = delta.max(axis=0)
    eps = 1e-3
    x = np.column_stack(
        [
            p[ci, ki],
            last[ci, ki],
            prior[ci, ki],
            delta[ci, ki],
            (last - prior)[ci, ki],
            np.clip(np.log((p[ci, ki] + eps) / (prior[ci, ki] + eps)), -10, 10),
            np.log1p(base_count[ci, ki]),
            np.log1p(n3[ci]),
            p[ci, ki] / max_within[ci],
            np.log1p(global_count[ki]),
            breadth[ki],
            growing[ki],
            other_mean[ci, ki],
            max_delta[ki],
        ]
    ).astype(np.float32)
    if horizon_feature is not None:
        hcol = np.full((len(x), 1), np.log1p(horizon_feature) / np.log(11.0), dtype=np.float32)
        x = np.hstack([x, hcol])
    if not np.isfinite(x).all():
        raise RuntimeError("non-finite emergence feature")
    return x


def sample_training(item: dict, origin: int, horizon: int):
    idx, y = universe_and_labels(item, horizon)
    pos = idx[y == 1]
    neg = idx[y == 0]
    n_take = min(len(neg), max(2000, 5 * len(pos)))
    rng = np.random.default_rng(SEED + origin * 37 + horizon * 1009)
    neg_take = neg if n_take == len(neg) else rng.choice(neg, size=n_take, replace=False)
    chosen = np.concatenate([pos, neg_take])
    chosen_y = np.concatenate(
        [np.ones(len(pos), dtype=np.uint8), np.zeros(len(neg_take), dtype=np.uint8)]
    )
    order = rng.permutation(len(chosen))
    chosen, chosen_y = chosen[order], chosen_y[order]
    x = feature_matrix(item, chosen)
    neg_weight = len(neg) / max(len(neg_take), 1)
    weight = np.where(chosen_y == 1, 1.0, neg_weight).astype(np.float64)
    return x, chosen_y, weight, {
        "origin": origin,
        "horizon": horizon,
        "eligible_full": len(idx),
        "positive_full": len(pos),
        "negative_full": len(neg),
        "positive_sampled": len(pos),
        "negative_sampled": len(neg_take),
        "negative_inclusion_probability": len(neg_take) / max(len(neg), 1),
        "negative_inverse_probability_weight": neg_weight,
        "seed": SEED + origin * 37 + horizon * 1009,
    }


def build_sample_cache(panel: dict):
    cache = {}
    audit = []
    for origin, item in panel["origins"].items():
        for horizon in HORIZONS:
            if horizon not in item["topic"]["future"]:
                continue
            x, y, w, row = sample_training(item, int(origin), horizon)
            cache[(int(origin), horizon)] = (x, y, w)
            audit.append(row)
    return cache, audit


def training_data(cache: dict, cutoff: int, target_h: int, scope: str):
    arrays = []
    used = []
    for (origin, horizon), (x, y, w) in sorted(cache.items()):
        if origin + horizon > cutoff:
            continue
        if scope == "specific" and horizon != target_h:
            continue
        xx = x
        if scope == "shared":
            hcol = np.full((len(x), 1), np.log1p(horizon) / np.log(11.0), dtype=np.float32)
            xx = np.hstack([x, hcol])
        arrays.append((xx, y, w))
        used.append((origin, horizon, len(y)))
    if not arrays:
        return np.empty((0, 15 if scope == "shared" else 14)), np.empty(0), np.empty(0), used
    x = np.vstack([a[0] for a in arrays])
    y = np.concatenate([a[1] for a in arrays])
    w = np.concatenate([a[2] for a in arrays])
    if len(w):
        w /= max(w.mean(), 1e-12)
    return x, y, w, used


def fit_model(x, y, w, learner: str):
    if len(y) == 0 or len(np.unique(y)) < 2:
        value = float(np.average(y, weights=w)) if len(y) else 0.0
        return {"kind": "constant", "value": value}
    if learner == "logit":
        scaler = StandardScaler()
        xx = scaler.fit_transform(x)
        model = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=300, random_state=SEED
        )
        model.fit(xx, y, sample_weight=w)
        return {"kind": "logit", "model": model, "scaler": scaler}
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=80,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=SEED,
    )
    model.fit(x, y, sample_weight=w)
    return {"kind": "hgb", "model": model}


def predict_score(bundle: dict, x: np.ndarray) -> np.ndarray:
    if bundle["kind"] == "constant":
        return np.full(len(x), bundle["value"], dtype=np.float64)
    xx = bundle["scaler"].transform(x) if bundle["kind"] == "logit" else x
    return bundle["model"].predict_proba(xx)[:, 1]


def counts_at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    yy = y.astype(bool)
    tp = int(np.sum(pred & yy))
    fp = int(np.sum(pred & ~yy))
    tn = int(np.sum(~pred & ~yy))
    fn = int(np.sum(~pred & yy))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else 0.0
    ba = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {
        "balanced_accuracy": ba,
        "precision": precision,
        "recall": recall,
        "true_events": tp + fn,
        "predicted_events": tp + fp,
        "eligible_samples": len(y),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def inner_origins(outer_origin: int, horizon: int) -> list[int]:
    start = max(1990 + horizon, outer_origin - horizon - 2)
    return list(range(start, outer_origin - horizon + 1))


def candidate_id(learner: str, scope: str, threshold: float) -> str:
    return f"{learner}_{scope}_thr{threshold:g}"


def needed_validation_origins() -> dict[int, list[int]]:
    result = {}
    for h in HORIZONS:
        values = set()
        for outer in OUTER[h] + [2024]:
            values.update(inner_origins(outer, h))
        result[h] = sorted(values)
    return result


def run_inner_validations(panel: dict, cache: dict):
    rows = []
    audit = []
    for h, origins in needed_validation_origins().items():
        for v in origins:
            item = panel["origins"][v]
            idx, y = universe_and_labels(item, h)
            x_base = feature_matrix(item, idx)
            for scope in ("specific", "shared"):
                x_eval = (
                    np.hstack(
                        [
                            x_base,
                            np.full(
                                (len(x_base), 1),
                                np.log1p(h) / np.log(11.0),
                                dtype=np.float32,
                            ),
                        ]
                    )
                    if scope == "shared"
                    else x_base
                )
                for learner in ("logit", "hgb"):
                    x, yy, w, used = training_data(cache, v, h, scope)
                    bundle = fit_model(x, yy, w, learner)
                    score = predict_score(bundle, x_eval)
                    audit.append(
                        {
                            "stage": "inner_validation_fit",
                            "fit_cutoff": v,
                            "evaluation_origin": v,
                            "target_horizon": h,
                            "learner": learner,
                            "scope": scope,
                            "training_rows": len(yy),
                            "training_positive": int(yy.sum()),
                            "training_origin_horizon_cells": len(used),
                            "max_training_target_end": max((u + q for u, q, _ in used), default=None),
                            "maturity_pass": all(u + q <= v for u, q, _ in used),
                        }
                    )
                    for threshold in THRESHOLDS:
                        row = {
                            "validation_origin": v,
                            "horizon": h,
                            "learner": learner,
                            "scope": scope,
                            "threshold": threshold,
                            "config_id": candidate_id(learner, scope, threshold),
                        }
                        row.update(counts_at_threshold(y, score, threshold))
                        rows.append(row)
            print(f"inner validation h={h} origin={v} n={len(y)}", flush=True)
    return rows, audit


def select_config(inner_rows: list[dict], origins: list[int], horizon: int):
    candidates = {}
    for row in inner_rows:
        if row["horizon"] != horizon or row["validation_origin"] not in origins:
            continue
        candidates.setdefault(row["config_id"], []).append(row)
    ranked = []
    for config, values in candidates.items():
        found_origins = sorted(r["validation_origin"] for r in values)
        ba_values = [r["balanced_accuracy"] for r in values]
        if found_origins != sorted(origins) or any(value is None or not np.isfinite(value) for value in ba_values):
            continue
        ranked.append(
            {
                "config_id": config,
                "learner": values[0]["learner"],
                "scope": values[0]["scope"],
                "threshold": values[0]["threshold"],
                "mean_balanced_accuracy": float(np.mean(ba_values)),
                "mean_precision": float(np.mean([r["precision"] for r in values])),
                "mean_recall": float(np.mean([r["recall"] for r in values if r["recall"] is not None])),
                "valid_inner_origins": len(ba_values),
                "requested_inner_origins": origins,
                "all_requested_inner_ba_finite": True,
            }
        )
    ranked.sort(
        key=lambda r: (
            -r["mean_balanced_accuracy"],
            -r["mean_precision"],
            -r["mean_recall"],
            r["config_id"],
        )
    )
    return (ranked[0] if ranked else None), ranked


def fit_selected(cache: dict, cutoff: int, horizon: int, selected: dict):
    x, y, w, used = training_data(cache, cutoff, horizon, selected["scope"])
    bundle = fit_model(x, y, w, selected["learner"])
    audit = {
        "stage": "selected_fit",
        "fit_cutoff": cutoff,
        "evaluation_origin": cutoff,
        "target_horizon": horizon,
        "learner": selected["learner"],
        "scope": selected["scope"],
        "training_rows": len(y),
        "training_positive": int(y.sum()),
        "training_origin_horizon_cells": len(used),
        "max_training_target_end": max((u + q for u, q, _ in used), default=None),
        "maturity_pass": all(u + q <= cutoff for u, q, _ in used),
    }
    return bundle, audit


def code_names(codes: list[str]) -> tuple[dict[str, str], dict]:
    """Optional display labels; code identifiers remain the model inputs."""
    mapping = {}
    if CODE_NAMES_FILE is not None:
        mapping = json.loads(CODE_NAMES_FILE.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise ValueError("code names must be a JSON object mapping codes to strings")
    return {code: str(mapping.get(code, code)) for code in codes}, {
        "display_labels_supplied": CODE_NAMES_FILE is not None,
        "use": "display names only; not model inputs",
    }


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    design, panel_manifest, panel, panel_hash = load_inputs()
    boundary_audit = audit_float_boundaries(panel)
    dump_json(OUT / "integer_boundary_audit.json", boundary_audit)
    cache, sample_audit = build_sample_cache(panel)
    write_csv(OUT / "training_sample_audit.csv", sample_audit)

    inner_rows, time_audit = run_inner_validations(panel, cache)
    write_csv(OUT / "inner_selection_metrics.csv", inner_rows)

    selections = []
    backtests = []
    top5_metrics = []
    top5_items = []
    eval_dir = OUT / "eval_predictions"
    eval_dir.mkdir(parents=True, exist_ok=True)
    names, name_sources = code_names(panel["codes"])
    dump_json(OUT / "code_name_sources.json", name_sources)
    for h in HORIZONS:
        for origin in OUTER[h]:
            origins = inner_origins(origin, h)
            selected, ranking = select_config(inner_rows, origins, h)
            if selected is None:
                backtests.append(
                    {
                        "origin": origin,
                        "horizon": h,
                        "target_year": origin + h,
                        "status": "no_valid_inner_BA",
                        "config_id": "",
                        "learner": "",
                        "scope": "",
                        "threshold": "",
                        **{k: "" for k in (
                            "balanced_accuracy", "precision", "recall", "true_events",
                            "predicted_events", "eligible_samples", "tp", "fp", "tn", "fn"
                        )},
                    }
                )
                continue
            selections.append(
                {
                    "stage": "outer",
                    "origin": origin,
                    "horizon": h,
                    "inner_origins": origins,
                    "selected": selected,
                    "ranking": ranking,
                }
            )
            bundle, audit = fit_selected(cache, origin, h, selected)
            time_audit.append(audit)
            idx, y = universe_and_labels(panel["origins"][origin], h)
            x = feature_matrix(
                panel["origins"][origin],
                idx,
                h if selected["scope"] == "shared" else None,
            )
            score = predict_score(bundle, x)
            metrics = counts_at_threshold(y, score, selected["threshold"])
            np.savez_compressed(
                eval_dir / f"origin_{origin}_h{h}.npz",
                flat_idx=idx.astype(np.int64),
                y=y.astype(np.uint8),
                score=score.astype(np.float32),
                threshold=np.asarray(selected["threshold"], dtype=np.float64),
                config_id=np.asarray(selected["config_id"]),
            )
            order5 = np.lexsort((idx, -score))[:5]
            top_hits = y[order5]
            terms = panel["origins"][origin]["topic"]["terms"]
            width = panel["origins"][origin]["topic"]["p"].shape[1]
            hit_labels = []
            for rank, loc in enumerate(order5, 1):
                flat = int(idx[loc])
                ci, ki = divmod(flat, width)
                item_row = {
                    "origin": origin,
                    "horizon": h,
                    "target_year": origin + h,
                    "rank": rank,
                    "flat_idx": flat,
                    "code": panel["codes"][ci],
                    "discipline_name": names[panel["codes"][ci]],
                    "keyword": terms[ki],
                    "score": float(score[loc]),
                    "is_true_event": int(y[loc]),
                    "config_id": selected["config_id"],
                }
                top5_items.append(item_row)
                if y[loc]:
                    hit_labels.append(f"{panel['codes'][ci]}|{terms[ki]}")
            top5_metrics.append(
                {
                    "origin": origin,
                    "horizon": h,
                    "target_year": origin + h,
                    "config_id": selected["config_id"],
                    "hits_at_5": int(top_hits.sum()),
                    "precision_at_5": float(top_hits.mean()),
                    "hit_code_keywords": json.dumps(hit_labels, ensure_ascii=False),
                    "ranking_rule": "score descending, flat_idx ascending tie-break",
                    "evaluation_mask_n": len(y),
                }
            )
            row = {
                "origin": origin,
                "horizon": h,
                "target_year": origin + h,
                "status": "evaluated",
                "config_id": selected["config_id"],
                "learner": selected["learner"],
                "scope": selected["scope"],
                "threshold": selected["threshold"],
            }
            row.update(metrics)
            backtests.append(row)
            print(
                f"outer h={h} origin={origin} config={selected['config_id']} BA={metrics['balanced_accuracy']}",
                flush=True,
            )

    future_rows = []
    final_models = {}
    for h in HORIZONS:
        origins = inner_origins(2024, h)
        selected, ranking = select_config(inner_rows, origins, h)
        if selected is None:
            raise RuntimeError(f"no final selection for h={h}")
        selections.append(
            {
                "stage": "final",
                "origin": 2024,
                "horizon": h,
                "inner_origins": origins,
                "selected": selected,
                "ranking": ranking,
            }
        )
        bundle, audit = fit_selected(cache, 2024, h, selected)
        time_audit.append(audit)
        final_models[h] = {
            "selected": selected,
            "feature_names": FEATURE_NAMES + (["log1p_h_scaled"] if selected["scope"] == "shared" else []),
            "model_bundle": bundle,
        }
        item = panel["origins"][2024]
        idx, _ = universe_and_labels(item, None)
        x = feature_matrix(item, idx, h if selected["scope"] == "shared" else None)
        score = predict_score(bundle, x)
        order = np.lexsort((idx, -score))[:20]
        p = np.asarray(item["topic"]["p"])
        n3 = np.asarray(item["n3"])
        terms = item["topic"]["terms"]
        k = p.shape[1]
        for rank, loc in enumerate(order, 1):
            flat = int(idx[loc])
            ci, ki = divmod(flat, k)
            value = float(score[loc])
            future_rows.append(
                {
                    "origin": 2024,
                    "horizon": h,
                    "target_year": 2024 + h,
                    "rank": rank,
                    "code": panel["codes"][ci],
                    "discipline_name": names[panel["codes"][ci]],
                    "keyword": terms[ki],
                    "score": value,
                    "score_is_calibrated_probability": False,
                    "above_selected_threshold": value >= selected["threshold"],
                    "selected_threshold": selected["threshold"],
                    "current_share": float(p[ci, ki]),
                    "current_keyword_support": float(p[ci, ki] * n3[ci]),
                    "current_code_n3": float(n3[ci]),
                    "base_zero_entry": bool(p[ci, ki] == 0),
                    "config_id": selected["config_id"],
                }
            )

    write_csv(OUT / "backtest_metrics.csv", backtests)
    write_csv(OUT / "top5_metrics.csv", top5_metrics)
    write_csv(OUT / "backtest_top5_items.csv", top5_items)
    write_csv(OUT / "time_audit.csv", time_audit)
    write_csv(OUT / "future_candidates.csv", future_rows)
    dump_json(OUT / "candidate_selection.json", selections)
    with gzip.open(OUT / "final_models.pkl.gz", "wb", compresslevel=5) as handle:
        pickle.dump(final_models, handle, pickle.HIGHEST_PROTOCOL)

    eval_recompute_pass = True
    top5_recompute_pass = True
    for row in backtests:
        if row["status"] != "evaluated":
            continue
        payload = np.load(eval_dir / f"origin_{row['origin']}_h{row['horizon']}.npz")
        recomputed = counts_at_threshold(
            payload["y"], payload["score"], float(payload["threshold"])
        )
        for field in ("true_events", "predicted_events", "eligible_samples", "tp", "fp", "tn", "fn"):
            eval_recompute_pass &= recomputed[field] == row[field]
        for field in ("balanced_accuracy", "precision", "recall"):
            eval_recompute_pass &= bool(np.isclose(recomputed[field], row[field]))
        order = np.lexsort((payload["flat_idx"], -payload["score"]))[:5]
        stored = next(
            value for value in top5_metrics
            if value["origin"] == row["origin"] and value["horizon"] == row["horizon"]
        )
        top5_recompute_pass &= int(payload["y"][order].sum()) == stored["hits_at_5"]

    checks = {
        "panel_hash_matches_manifest": panel_hash == panel_manifest["output_sha256"],
        "feature_count_is_14": len(FEATURE_NAMES) == 14,
        "all_training_rows_mature": all(row["maturity_pass"] for row in time_audit),
        "outer_origin_sets_exact": {
            str(h): sorted(r["origin"] for r in backtests if r["horizon"] == h) == OUTER[h]
            for h in HORIZONS
        },
        "all_outer_evaluations_full": all(
            row["eligible_samples"] == row["tp"] + row["fp"] + row["tn"] + row["fn"]
            for row in backtests
            if row["status"] == "evaluated"
        ),
        "all_17_outer_prediction_payloads_saved": len(list(eval_dir.glob("origin_*_h*.npz"))) == 17,
        "npz_metrics_independently_recompute": eval_recompute_pass,
        "top5_hits_independently_recompute": top5_recompute_pass,
        "all_selected_configs_have_complete_inner_ba": all(
            entry["selected"]["all_requested_inner_ba_finite"] for entry in selections
        ),
        "future_at_least_five_each": {
            str(h): sum(r["horizon"] == h for r in future_rows) >= 5 for h in HORIZONS
        },
        "future_scores_declared_uncalibrated": all(
            not row["score_is_calibrated_probability"] for row in future_rows
        ),
        "future_uses_origin_2024": all(row["origin"] == 2024 for row in future_rows),
        "future_names_resolved": all(row["discipline_name"] != "名称缺失" for row in future_rows),
        "h1_known_years_disclosed": True,
        "integer_boundary_rule_used": boundary_audit["integer_rule_is_current"],
    }
    verification = {"status": "PASS" if all(
        value if isinstance(value, bool) else all(value.values())
        for value in checks.values()
    ) else "FAIL", "checks": checks}
    dump_json(OUT / "verification.json", verification)
    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "seconds": time.time() - started,
        "panel_sha256": panel_hash,
        "design_manifest_sha256": sha256(DESIGN_PATH),
        "feature_names": FEATURE_NAMES,
        "code_name_sources": name_sources,
        "outputs": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(OUT.iterdir())
            if path.is_file() and path.name != "run_manifest.json"
        },
        "evaluation_prediction_files": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(eval_dir.glob("*.npz"))
        },
        "verification_status": verification["status"],
        "notes": [
            "No relation attraction/repulsion features were used.",
            "Future fields were used only to construct mature historical labels.",
            "Event thresholds were evaluated by exact int64 cross-multiplication to avoid float32 boundary misclassification; threshold values were unchanged.",
            "Negative samples were inverse-probability weighted; validation and outer tests used all eligible rows.",
            "Scores are ranking scores and are not claimed to be calibrated probabilities."
        ],
    }
    dump_json(OUT / "run_manifest.json", manifest)
    print(json.dumps({"status": verification["status"], "seconds": manifest["seconds"]}, ensure_ascii=False))


def cli() -> None:
    global OUT, PANEL_PATH, PANEL_MANIFEST, DESIGN_PATH, CODE_NAMES_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--code-names", type=Path, default=None,
                        help="optional JSON mapping from F-codes to display names")
    args = parser.parse_args()
    PANEL_PATH = args.panel.resolve()
    PANEL_MANIFEST = PANEL_PATH.parent / "panel_manifest.json"
    OUT = args.output.resolve()
    DESIGN_PATH = OUT / "design_manifest.json"
    CODE_NAMES_FILE = args.code_names.resolve() if args.code_names else None
    if OUT.exists() and any(OUT.iterdir()):
        parser.error("output directory is not empty; use a fresh --output directory")
    if not PANEL_PATH.is_file() or not PANEL_MANIFEST.is_file():
        parser.error("build the panel and its panel_manifest.json first")
    OUT.mkdir(parents=True, exist_ok=True)
    dump_json(DESIGN_PATH, {
        "panel": {"expected_sha256": sha256(PANEL_PATH)},
        "horizons": list(HORIZONS),
        "base_seed": SEED,
        "selection": "latest three mature development origins; binary BA, precision, recall",
    })
    with threadpool_limits(limits=1):
        main()


if __name__ == "__main__":
    cli()
