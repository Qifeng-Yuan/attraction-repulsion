"""Fit attraction and repulsion classifiers for 1/5/10-year forecasts.

Training and development labels must end by each forecast origin.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import pickle
import time
import traceback
from pathlib import Path
from typing import Any

# Use one OpenMP worker for Windows HGB compatibility and cap BLAS at two.
os.environ["OMP_NUM_THREADS"] = "1"
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "2"

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DEFAULT_PANEL = ROOT / "runs" / "panel.pkl.gz"
DEFAULT_OUTPUT = ROOT / "runs" / "relations"
SEED = 20260909
TRAIN_START = 1990
CLASS_THRESHOLD = 0.03
HORIZONS = (1, 5, 10)
BRANCHES = ("plus", "minus")
THRESHOLDS = np.asarray([.01, .025, .05, .10, .15, .20, .30, .40, .50,
                         .60, .70, .80, .90, .95])
OUTER = {1: tuple(range(2018, 2024)), 5: tuple(range(2014, 2020)),
         10: tuple(range(2010, 2015))}
MODEL_ID = "attraction_repulsion"
_SCORE_CACHE: dict[tuple, tuple[np.ndarray, dict[str, Any]]] = {}


def _json_default(x: Any) -> Any:
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def atomic_pickle(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=4) as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def read_pickle(path: Path) -> Any:
    with gzip.open(path, "rb") as f: return pickle.load(f)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def lookup(d: dict, key: int) -> Any:
    if key in d: return d[key]
    if str(key) in d: return d[str(key)]
    raise KeyError(key)


def classes(delta: np.ndarray) -> np.ndarray:
    return np.where(delta > CLASS_THRESHOLD, 1,
                    np.where(delta < -CLASS_THRESHOLD, -1, 0)).astype(np.int8)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y, pred, labels=[-1, 0, 1])
    n = cm.sum(1); pn = cm.sum(0)
    recall = np.divide(np.diag(cm), n, out=np.full(3, np.nan), where=n > 0)
    precision = np.divide(np.diag(cm), pn, out=np.zeros(3), where=pn > 0)
    out = {"n": int(len(y)), "balanced_accuracy":
           float(np.mean(recall)) if np.all(n > 0) else np.nan,
           "accuracy": float(np.trace(cm) / max(len(y), 1))}
    for k, name in enumerate(("separation", "stable", "convergence")):
        out[name + "_n"] = int(n[k]); out[name + "_predicted_n"] = int(pn[k])
        out[name + "_recall"] = float(recall[k]); out[name + "_precision"] = float(precision[k])
    return out


def decide(plus: np.ndarray, minus: np.ndarray, tp: float, tm: float):
    ep = (plus - tp) / max(1 - tp, 1e-9)
    em = (minus - tm) / max(1 - tm, 1e-9)
    pred = np.where((ep < 0) & (em < 0), 0, np.where(ep >= em, 1, -1)).astype(np.int8)
    return pred, (ep >= 0) & (em >= 0), np.abs(ep - em)


def geometry_grid(limit: int | None = None) -> list[dict[str, Any]]:
    trends = [("alpha", x) for x in (.75, .90, 1.0)] + [("rolling", x) for x in (.5, 1., 2.)]
    settings = [(.005, 20., .03), (.010, 40., .05), (.020, 60., .08), (.030, 100., .12)]
    result = []
    for mode, value in trends:
        for deadzone, rho, rowfloor in settings:
            result.append({"id": f"{mode}_{value:g}_d{deadzone:g}_r{rho:g}_f{rowfloor:g}",
                           "trend_mode": mode, "trend_value": value,
                           "deadzone": deadzone, "rho": rho, "rowfloor": rowfloor})
    return result if limit is None else result[:limit]


LEARNERS = [
    {"id": "shared_lda_equal", "kind": "lda", "time_half_life": None, "training_scope": "shared"},
    {"id": "shared_logit_c03_h6", "kind": "logit", "C": .3, "time_half_life": 6., "training_scope": "shared"},
    {"id": "shared_logit_c1_h6", "kind": "logit", "C": 1., "time_half_life": 6., "training_scope": "shared"},
    {"id": "shared_hgb_h6", "kind": "hgb", "time_half_life": 6., "training_scope": "shared"},
    {"id": "specific_lda_equal", "kind": "lda", "time_half_life": None, "training_scope": "specific"},
    {"id": "specific_logit_c1_h6", "kind": "logit", "C": 1., "time_half_life": 6., "training_scope": "specific"},
]


def validate_panel(panel: dict) -> tuple[int, tuple[int, ...]]:
    for key in ("codes", "pair_i", "pair_j", "origins"):
        if key not in panel: raise KeyError(f"panel missing {key}")
    pi, pj = np.asarray(panel["pair_i"]), np.asarray(panel["pair_j"])
    if pi.ndim != 1 or pj.shape != pi.shape: raise ValueError("unaligned pair indices")
    years = tuple(sorted(int(k) for k in panel["origins"]))
    if not years or years[0] > TRAIN_START or years[-1] < 2024:
        raise ValueError(f"origin coverage {years[:1]}..{years[-1:]}, expected 1990..2024")
    p = len(pi); c = len(panel["codes"])
    for T in years:
        rec = lookup(panel["origins"], T)
        for key in ("annual_R", "rolling_R"):
            x = np.asarray(rec[key]); yy = np.asarray(rec[key.replace("R", "years")])
            if x.ndim != 3 or x.shape[1:] != (c, c) or len(x) != len(yy):
                raise ValueError(f"bad {key} at {T}")
            if len(yy) == 0 or yy[-1] != T or np.any(np.diff(yy) <= 0):
                raise ValueError(f"{key} chronology failed at {T}")
        valid = np.asarray(rec["rolling_valid"])
        if valid.shape != (len(rec["rolling_years"]), c): raise ValueError(f"bad rolling_valid at {T}")
        z = np.asarray(rec["z"]); u = np.asarray(rec["u"])
        if z.shape != u.shape or z.shape[0] != c: raise ValueError(f"bad z/u at {T}")
        if z.shape[1] != len(rec["idf"]): raise ValueError(f"bad idf at {T}")
        un = np.linalg.norm(u, axis=1); zn = np.linalg.norm(z, axis=1)
        if not np.allclose(un[zn > 0], 1., atol=2e-4): raise ValueError(f"u not normalized at {T}")
        for key in ("eligible", "base_R"):
            if np.asarray(rec[key]).shape != (p,): raise ValueError(f"bad {key} at {T}")
        if np.asarray(rec["n3"]).shape != (c,): raise ValueError(f"bad n3 at {T}")
        for h in HORIZONS:
            counter = lookup(rec["counter"], h)
            if np.asarray(counter["values"]).shape != (p, 5): raise ValueError(f"bad counter T{T} h{h}")
            if not np.isfinite(counter["values"]).all(): raise ValueError(f"nonfinite counter T{T} h{h}")
            if T + h <= 2024:
                lab = lookup(rec["labels"], h)
                if np.asarray(lab["delta"]).shape != (p,) or np.asarray(lab["mask"]).shape != (p,):
                    raise ValueError(f"bad label T{T} h{h}")
    return p, years


def _history_pad(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == 0: raise ValueError("empty history")
    if len(x) >= n: return x[-n:]
    return np.concatenate([np.repeat(x[:1], n - len(x), axis=0), x], axis=0)


def _validity_pad(x: np.ndarray, n: int) -> np.ndarray:
    """Mark padded history as unavailable."""
    if len(x) >= n: return x[-n:]
    return np.concatenate([np.zeros((n - len(x),) + x.shape[1:], dtype=bool), x], axis=0)


def make_features(panel: dict, T: int, h: int, cfg: dict, branch: str) -> np.ndarray:
    rec = lookup(panel["origins"], T); pi = np.asarray(panel["pair_i"]); pj = np.asarray(panel["pair_j"])
    c = len(panel["codes"]); annual = _history_pad(np.asarray(rec["annual_R"], dtype=np.float32), 6)
    rolling_raw = np.asarray(rec["rolling_R"], dtype=np.float32)
    valid_raw = np.asarray(rec["rolling_valid"], dtype=bool)
    rolling = _history_pad(rolling_raw, 6); valid = _validity_pad(valid_raw, 6)
    r = np.asarray(rec["base_R"], dtype=np.float32); R = np.zeros((c, c), np.float32)
    R[pi, pj] = r; R[pj, pi] = r
    # Differences require valid vectors for both disciplines at both endpoints.
    pair_valid = valid[:, :, None] & valid[:, None, :]
    droll = np.diff(rolling, axis=0)
    transition_valid = pair_valid[1:] & pair_valid[:-1]
    droll = np.where(transition_valid, droll, 0.)
    d1 = droll[-1]; d2 = droll[-2]
    if cfg["trend_mode"] == "alpha":
        a = cfg["trend_value"]; D = a * d1 + (1 - a) * d2
    else:
        ages = np.arange(len(droll) - 1, -1, -1)
        w = np.exp2(-ages / cfg["trend_value"])
        den = np.tensordot(w, transition_valid.astype(np.float32), axes=(0, 0))
        D = np.divide(np.tensordot(w, droll, axes=(0, 0)), den,
                      out=np.zeros_like(d1), where=den > 0)
    n3 = np.asarray(rec["n3"], dtype=np.float32)
    relc = n3 / (n3 + cfg["rho"]); reliability = np.sqrt(relc[:, None] * relc[None, :])
    sign = 1. if branch == "plus" else -1.
    W = R * np.maximum(sign * D - cfg["deadzone"], 0) * reliability
    np.fill_diagonal(W, 0); strength_raw = W.sum(1)
    W = W / np.maximum(strength_raw[:, None], cfg["rowfloor"])
    strength = W.sum(1)
    z = np.asarray(rec["z"], dtype=np.float32); zn = np.linalg.norm(z, axis=1)
    u = np.asarray(rec["u"], dtype=np.float32)
    # Compute forces on weighted states and cosine derivatives on unit vectors.
    force = sign * (W @ z - strength[:, None] * z)
    du = (force - u * np.sum(u * force, axis=1, keepdims=True)) / np.maximum(zn[:, None], 1e-8)
    J = du @ u.T + u @ du.T
    common = W @ W.T; reciprocal = W @ W
    A = R.copy(); np.fill_diagonal(A, 0); degree = A.sum(1); cn = (A @ A) / max(c, 1)
    minimum = np.minimum(n3[pi], n3[pj]); maximum = np.maximum(n3[pi], n3[pj])
    hist_pairs = annual[:, pi, pj]; roll_pairs = rolling[:, pi, pj]
    avail = pair_valid[:, pi, pj].astype(np.float32)
    roll_mean = np.divide((roll_pairs * avail).sum(0), avail.sum(0),
                          out=np.zeros(len(pi), np.float32), where=avail.sum(0) > 0)
    roll_var = np.divide(((roll_pairs - roll_mean) ** 2 * avail).sum(0), avail.sum(0),
                         out=np.zeros(len(pi), np.float32), where=avail.sum(0) > 0)
    counter = np.asarray(lookup(rec["counter"], h)["values"], dtype=np.float32)
    known = np.asarray(lookup(rec["known_fraction"], h), dtype=np.float32)
    if known.ndim == 0: known = np.full(c, float(known), np.float32)
    same_parent = np.asarray([str(panel["codes"][a])[:3] == str(panel["codes"][b])[:3]
                              for a, b in zip(pi, pj)], dtype=np.float32)
    base = np.column_stack([
        r, d1[pi, pj], d2[pi, pj], D[pi, pj], hist_pairs[-1],
        roll_mean, np.sqrt(roll_var), avail.sum(0), avail[-1],
        transition_valid[:, pi, pj].sum(0), cn[pi, pj],
        np.minimum(degree[pi], degree[pj]) / max(c, 1),
        np.abs(degree[pi] - degree[pj]) / max(c, 1), np.log1p(minimum),
        np.log1p(maximum), np.abs(np.log1p(n3[pi]) - np.log1p(n3[pj])), same_parent,
        minimum / (minimum + cfg["rho"]), np.minimum(known[pi], known[pj]),
        np.abs(known[pi] - known[pj]), np.full(len(pi), np.log1p(h), np.float32),
        np.full(len(pi), h / 10., np.float32), counter,
    ])
    own = np.column_stack([
        sign * J[pi, pj], np.maximum(sign * d1[pi, pj], 0),
        np.maximum(sign * d2[pi, pj], 0), np.maximum(sign * D[pi, pj], 0),
        common[pi, pj], reciprocal[pi, pj], np.minimum(strength[pi], strength[pj]),
        np.abs(strength[pi] - strength[pj]), np.maximum(sign * counter, 0),
    ])
    out = np.column_stack([base, own]).astype(np.float32)
    if not np.isfinite(out).all(): raise ValueError(f"nonfinite features T{T} h{h} {cfg['id']} {branch}")
    return out


def development_origins(T: int, h: int) -> list[int]:
    years = list(range(max(TRAIN_START + h, T - h - 2), T - h + 1))
    if not years: raise ValueError(f"no mature dev origins T={T} h={h}")
    return years


def training_origins(T: int, h: int) -> list[int]:
    years = list(range(TRAIN_START, T - h + 1))
    if not years or any(v + h > T for v in years): raise AssertionError("training maturity failed")
    return years


def training_arrays(panel: dict, T: int, h: int, cfg: dict, branch: str, scope: str):
    xs, ys, origins = [], [], []
    train_horizons = HORIZONS if scope == "shared" else (h,)
    for q in train_horizons:
        for v in training_origins(T, q):
            assert v + q <= T
            rec = lookup(panel["origins"], v); lab = lookup(rec["labels"], q)
            mask = np.asarray(lab["mask"], bool)
            if not mask.any(): continue
            delta = np.asarray(lab["delta"], np.float32)[mask]
            xs.append(make_features(panel, v, q, cfg, branch)[mask])
            ys.append(delta > CLASS_THRESHOLD if branch == "plus" else delta < -CLASS_THRESHOLD)
            origins.append(np.full(mask.sum(), v, dtype=np.int16))
    if not xs: return None, np.empty(0, np.uint8), np.empty(0, np.int16)
    return np.concatenate(xs), np.concatenate(ys).astype(np.uint8), np.concatenate(origins)


def fit_score(panel: dict, fit_T: int, pred_T: int, h: int, cfg: dict,
              branch: str, learner: dict) -> tuple[np.ndarray, dict[str, Any]]:
    cache_key = (fit_T, pred_T, h, cfg["id"], branch, learner["id"])
    if cache_key in _SCORE_CACHE: return _SCORE_CACHE[cache_key]
    x, y, origin = training_arrays(panel, fit_T, h, cfg, branch, learner["training_scope"])
    px = make_features(panel, pred_T, h, cfg, branch)
    if len(y) == 0:
        result = (np.full(len(px), .5, dtype=np.float32), {"n_train": 0, "negative_n": 0,
            "positive_n": 0, "fit_state": "constant_empty_uninformative_prior",
            "training_scope": learner["training_scope"]})
        _SCORE_CACHE[cache_key] = result; return result
    counts = np.bincount(y, minlength=2); diag = {"n_train": int(len(y)),
        "negative_n": int(counts[0]), "positive_n": int(counts[1]),
        "train_first": int(origin.min()), "train_last": int(origin.max()),
        "training_scope": learner["training_scope"]}
    unique = np.unique(y)
    if len(unique) == 1:
        diag["fit_state"] = "constant_single_class_smoothed_prior"
        prior = float((counts[1] + 1.) / (len(y) + 2.))
        result = (np.full(len(px), prior, dtype=np.float32), diag)
        _SCORE_CACHE[cache_key] = result; return result
    age = origin.max() - origin
    tw = np.ones(len(y)) if learner.get("time_half_life") is None else np.exp2(-age / learner["time_half_life"])
    cw = len(y) / (2 * counts.astype(float)); weight = tw * cw[y]
    kind = learner["kind"]
    if kind in ("lda", "logit"):
        scaler = StandardScaler(); xx = scaler.fit_transform(x); pxx = scaler.transform(px)
        if kind == "lda":
            model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[.5, .5])
            model.fit(xx, y)
        else:
            model = LogisticRegression(C=learner["C"], max_iter=300, solver="lbfgs", random_state=SEED)
            model.fit(xx, y, sample_weight=weight)
    else:
        model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15,
            learning_rate=.08, l2_regularization=10., min_samples_leaf=40,
            early_stopping=False, random_state=SEED)
        # Avoid worker-pipe failures in Windows HGB.
        with parallel_backend("sequential"):
            model.fit(x, y, sample_weight=weight)
        pxx = px
    col = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
    score = np.asarray(model.predict_proba(pxx)[:, col], dtype=np.float32)
    diag["fit_state"] = "fitted"
    result = (np.clip(score, 0, 1), diag)
    _SCORE_CACHE[cache_key] = result; return result


def best_thresholds(dev: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    best = None
    for tp in THRESHOLDS:
        for tm in THRESHOLDS:
            bas, conflicts = [], []
            for _, y, plus, minus in dev:
                pred, conflict, _ = decide(plus, minus, float(tp), float(tm))
                recalls = [float(np.mean(pred[y == label] == label)) if np.any(y == label) else np.nan
                           for label in (-1, 0, 1)]
                bas.append(float(np.mean(recalls)) if np.all(np.isfinite(recalls)) else np.nan)
                conflicts.append(float(conflict.mean()))
            finite = [x for x in bas if np.isfinite(x)]
            mean_ba = float(np.mean(finite)) if len(finite) == len(bas) and finite else -1.
            criterion = (mean_ba, -float(np.mean(conflicts)), -abs(float(tp)-.5)-abs(float(tm)-.5))
            if best is None or criterion > best[0]:
                best = (criterion, {"plus_threshold": float(tp), "minus_threshold": float(tm),
                    "development_mean_BA": mean_ba, "development_BA_by_origin": bas,
                    "development_conflict_by_origin": conflicts})
    return best[1]


def select(panel: dict, T: int, h: int, geometries: list[dict], logger: logging.Logger):
    dev_years = development_origins(T, h); candidate_rows = []
    # Select geometry with shared-horizon LDA.
    fixed = LEARNERS[0]; geo_rank = []
    for g in geometries:
        dev = []
        counts = []
        for v in dev_years:
            rec = lookup(panel["origins"], v); lab = lookup(rec["labels"], h); m = np.asarray(lab["mask"], bool)
            y = classes(np.asarray(lab["delta"])[m])
            sp, dp = fit_score(panel, v, v, h, g, "plus", fixed)
            sm, dm = fit_score(panel, v, v, h, g, "minus", fixed)
            dev.append((v, y, sp[m], sm[m])); counts.append(metrics(y, y))
        th = best_thresholds(dev)
        row = {"stage": "geometry", "prediction_origin": T, "horizon": h,
               "candidate_id": g["id"], "learner_id": fixed["id"],
               "development_origins": dev_years,
               "development_class_counts": [{k: x[k] for k in ("separation_n", "stable_n", "convergence_n")} for x in counts],
               **g, **th}
        candidate_rows.append(row); geo_rank.append((th["development_mean_BA"], g["id"], g))
    geo_rank.sort(key=lambda x: (-x[0], x[1])); chosen_g = geo_rank[0][2]
    # Select branch learners for the chosen geometry.
    model_rank = []; learner_devs = {}; learner_diags = {}
    for learner in LEARNERS:
        dev = []; fit_diags = []
        for v in dev_years:
            rec = lookup(panel["origins"], v); lab = lookup(rec["labels"], h); m = np.asarray(lab["mask"], bool)
            y = classes(np.asarray(lab["delta"])[m])
            sp, dp = fit_score(panel, v, v, h, chosen_g, "plus", learner)
            sm, dm = fit_score(panel, v, v, h, chosen_g, "minus", learner)
            dev.append((v, y, sp[m], sm[m])); fit_diags.append({"origin": v, "plus": dp, "minus": dm})
        learner_devs[learner["id"]] = dev; learner_diags[learner["id"]] = fit_diags
    for plus_learner in LEARNERS:
        for minus_learner in LEARNERS:
            pdev, mdev = learner_devs[plus_learner["id"]], learner_devs[minus_learner["id"]]
            dev = [(p[0], p[1], p[2], m[3]) for p, m in zip(pdev, mdev)]
            th = best_thresholds(dev)
            candidate_id = plus_learner["id"] + "__" + minus_learner["id"]
            row = {"stage": "independent_heads", "prediction_origin": T, "horizon": h,
                   "candidate_id": candidate_id, "geometry_id": chosen_g["id"],
                   "plus_learner_id": plus_learner["id"], "minus_learner_id": minus_learner["id"],
                   "development_origins": dev_years,
                   "plus_fit_diagnostics": learner_diags[plus_learner["id"]],
                   "minus_fit_diagnostics": learner_diags[minus_learner["id"]], **th}
            candidate_rows.append(row)
            model_rank.append((th["development_mean_BA"], candidate_id,
                               plus_learner, minus_learner, th))
    model_rank.sort(key=lambda x: (-x[0], x[1]))
    best_mean = model_rank[0][0]
    # Prefer matching learners when their development BA is within 0.005 of best.
    simple = [x for x in model_rank
              if x[2]["id"] == x[3]["id"] and x[0] >= best_mean - .005]
    chosen = sorted(simple, key=lambda x: (-x[0], x[1]))[0] if simple else model_rank[0]
    _, _, chosen_lp, chosen_lm, chosen_th = chosen
    logger.info("selected h%d T%d geometry=%s plus=%s minus=%s dev mean BA=%.4f", h, T,
                chosen_g["id"], chosen_lp["id"], chosen_lm["id"], chosen_th["development_mean_BA"])
    selection = {"prediction_origin": T, "horizon": h, "geometry": chosen_g,
                 "plus_learner": chosen_lp, "minus_learner": chosen_lm,
                 "development_origins": dev_years, **chosen_th,
                 "selection_rule": "max mean origin BA; if a same-learner pair is within 0.005, choose best such pair; deterministic id tie-break"}
    return selection, candidate_rows


def search_manifest(geometries: list[dict]) -> dict[str, Any]:
    return {"model_id": MODEL_ID, "created_before_outer_evaluation": True, "seed": SEED,
        "threads": 2, "class_definition": "delta < -.03, abs(delta)<=.03, delta > .03",
        "training_rule": "origins 1990..T-h inclusive; every label satisfies v+h<=T",
        "development_rule": "range(max(1990+h,T-h-2),T-h+1); optimize mean origin BA",
        "outer_origins": {str(k): list(v) for k, v in OUTER.items()}, "final_origin": 2024,
        "two_stage_rule": "24 geometries with shared-horizon LDA, then all 6x6 independently fitted plus/minus learner combinations for selected geometry",
        "learner_pair_rule": "prefer matching learners within 0.005 of the best mean development BA",
        "geometry_candidates": geometries, "learner_candidates": LEARNERS,
        "threshold_grid": THRESHOLDS.tolist(), "opposite_branch_feature_read": False,
        "shared_training_rule": "shared candidates pool q in {1,5,10}, each row only when v+q<=T; log(h) and h/10 are features",
        "empty_training_rule": "score 0.5; one-class training uses Laplace-smoothed empirical event prior",
        "missing_history_rule": "rolling differences require both nodes valid at both endpoints; otherwise difference=0; early numeric padding repeats earliest snapshot but validity padding is false, so availability length counts only real snapshots",
        "low_support_graph_rule": "past-only low-support neighbors are retained with rho reliability shrinkage; prediction/evaluation endpoints still use eligible support>=30",
        "alpha_formula": "alpha*dR(T,T-1)+(1-alpha)*dR(T-1,T-2) on adjacent rolling three-year cosine snapshots",
        "evaluation_rule": "evaluate outer labels after development selection",
        "selection_scope": "configured geometry, learner and threshold candidates"}


def run(panel_path: Path, output: Path, resume: bool, geometry_limit: int | None = None,
        target_override: dict[int, tuple[int, ...]] | None = None) -> None:
    _SCORE_CACHE.clear()
    output.mkdir(parents=True, exist_ok=True); started = time.time()
    logger = logging.getLogger("relation_model"); logger.handlers.clear(); logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(output / "relation_model.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(fmt); logger.addHandler(handler)
    geometries = geometry_grid(geometry_limit); manifest = search_manifest(geometries)
    atomic_json(output / "search_manifest.json", manifest)
    panel_hash = file_hash(panel_path); atomic_json(output / "status.json", {"status": "running", "panel_sha256": panel_hash})
    checkpoint_path = output / "checkpoint.pkl.gz"
    if checkpoint_path.exists():
        if not resume: raise RuntimeError("checkpoint exists; use --resume or a new output directory")
        state = read_pickle(checkpoint_path)
        if state.get("model_id") != MODEL_ID or state.get("panel_sha256") != panel_hash or state.get("search_manifest") != manifest:
            raise RuntimeError("checkpoint does not match panel or model configuration")
    else:
        state = {"model_id": MODEL_ID, "panel_sha256": panel_hash, "search_manifest": manifest,
                 "completed": [], "candidate_rows": [], "selections": [], "backtests": [],
                 "pair_rows": [], "prediction_arrays": {}}
    try:
        panel = read_pickle(panel_path); n_pairs, years = validate_panel(panel)
        targets = target_override or {h: OUTER[h] + (2024,) for h in HORIZONS}
        for h in targets:
            for T in targets[h]:
                key = [int(h), int(T)]
                if key in state["completed"]: logger.info("resume skip h%d T%d", h, T); continue
                selection, candidates = select(panel, T, h, geometries, logger)
                # Fit and evaluate the selected configuration.
                g = selection["geometry"]
                plus_learner, minus_learner = selection["plus_learner"], selection["minus_learner"]
                plus, dp = fit_score(panel, T, T, h, g, "plus", plus_learner)
                minus, dm = fit_score(panel, T, T, h, g, "minus", minus_learner)
                pred, conflict, margin = decide(plus, minus, selection["plus_threshold"], selection["minus_threshold"])
                selection["final_fit_diagnostics"] = {"plus": dp, "minus": dm}
                state["candidate_rows"].extend(candidates); state["selections"].append(selection)
                rec = lookup(panel["origins"], T); eligible = np.asarray(rec["eligible"], bool)
                state["prediction_arrays"][(h, T)] = {"plus_score": plus, "minus_score": minus,
                    "prediction": pred, "conflict": conflict, "decision_margin": margin}
                if T + h <= 2024:
                    lab = lookup(rec["labels"], h); mask = np.asarray(lab["mask"], bool)
                    met = metrics(classes(np.asarray(lab["delta"])[mask]), pred[mask])
                    state["backtests"].append({"horizon": h, "origin": T, "target_year": T+h,
                        "geometry_id": g["id"], "plus_learner_id": plus_learner["id"],
                        "minus_learner_id": minus_learner["id"],
                        "conflict_fraction": float(conflict[mask].mean()), **met})
                for q in np.flatnonzero(eligible):
                    state["pair_rows"].append({"horizon": h, "origin": T, "target_year": T+h,
                        "pair_index": int(q), "code1": panel["codes"][panel["pair_i"][q]],
                        "code2": panel["codes"][panel["pair_j"][q]], "base_R": float(rec["base_R"][q]),
                        "plus_score": float(plus[q]), "minus_score": float(minus[q]),
                        "prediction": int(pred[q]), "conflict": bool(conflict[q]),
                        "decision_margin": float(margin[q]), "geometry_id": g["id"],
                        "plus_learner_id": plus_learner["id"],
                        "minus_learner_id": minus_learner["id"]})
                state["completed"].append(key); atomic_pickle(checkpoint_path, state)
                atomic_json(output / "progress.json", {"completed": state["completed"], "updated_unix": time.time()})
        # Export tables from the checkpoint.
        pd.DataFrame(state["candidate_rows"]).to_csv(output / "candidate_dev_scores.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
        pd.DataFrame(state["selections"]).to_csv(output / "selected_parameters.csv", index=False, encoding="utf-8-sig")
        back = pd.DataFrame(state["backtests"]); back.to_csv(output / "outer_backtests.csv", index=False, encoding="utf-8-sig")
        pair = pd.DataFrame(state["pair_rows"]); pair.to_csv(output / "pair_predictions.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
        pair[pair["origin"] == 2024].to_csv(output / "future_2024_all_pairs.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
        atomic_pickle(output / "selected_predictions.pkl.gz", state["prediction_arrays"])
        means = []
        if len(back):
            for h, part in back.groupby("horizon"):
                means.append({"horizon": int(h), "origins": part["origin"].astype(int).tolist(),
                              "mean_balanced_accuracy": float(part["balanced_accuracy"].mean()),
                              "n_origins": int(len(part))})
        atomic_json(output / "outer_mean_BA.json", means)
        atomic_json(output / "status.json", {"status": "completed", "elapsed_seconds": time.time()-started,
            "panel_sha256": panel_hash, "pair_count": n_pairs, "origin_range": [years[0], years[-1]],
            "completed": state["completed"], "candidate_rows": len(state["candidate_rows"]),
            "outer_test_used_for_selection": False, "source_sha256": file_hash(Path(__file__))})
    except Exception as exc:
        logger.exception("Relation model run failed")
        atomic_json(output / "status.json", {"status": "failed", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(), "elapsed_seconds": time.time()-started})
        raise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--geometry-limit", type=int, default=None,
                   help="limit geometry candidates for quick checks (default: all 24)")
    args = p.parse_args()
    if args.geometry_limit is not None and not (1 <= args.geometry_limit <= 24): raise ValueError("geometry-limit must be 1..24")
    # Keep the OpenMP limit above; apply this limit to BLAS only.
    with threadpool_limits(limits=2, user_api="blas"):
        run(args.panel.resolve(), args.output.resolve(), args.resume, args.geometry_limit)


if __name__ == "__main__": main()
