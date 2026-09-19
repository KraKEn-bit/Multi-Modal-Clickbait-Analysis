"""
SECE v2 — physics-subset ensemble with NNLS meta-fusion.

Phase 1: subset experts + champion bases + NNLS
Phase 2: + df_ts3 frame for 3h, CB+MotionNN base, km-scaled residual path,
         champion-focused fusion + validation router
Phase 3: + multi-path fusion (champion/km/residual), 3h stacking-residual correction,
         48h LGB+PRC km-NNLS branch, position subset at 3h
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from pipeline_common import (
    H_LABELS,
    load_pipeline_data,
    track_errors_km,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT.parent / "results" / "sece_v2"
PREDS_DIR = OUT_DIR / "predictions"

GLOBAL_LR = 0.01
GLOBAL_EPOCHS_NN = 100
GLOBAL_BATCH = 128
DEG_TO_KM = np.array([111.0, 111.0])

SECE_SUBSETS_P1 = ["position", "motion", "physics", "environ", "trend", "full"]
SECE_SUBSETS_P2 = ["motion", "physics", "full"]
SECE_SUBSETS_P3_3H = ["position", "motion", "full"]
SECE_SUBSETS_P3_OTHER = ["motion", "physics", "full"]

SUBSET_KEYS = {
    "position": ["LAT", "LON", "dist_BoB", "lat_lon_int"],
    "motion": ["dLAT", "dLON", "geo_", "speed", "bearing", "accel", "curvature"],
    "physics": ["curvature", "bearing_stab", "speed_stab", "season", "bearing_con"],
    "environ": ["DIST2LAND", "LANDFALL", "NEWDELHI", "STORM_SPEED", "STORM_DIR"],
    "trend": ["trend", "momentum", "recurvature", "dist2land_rate", "accel"],
}

MOTION_FEAT_NAMES = [
    "dLAT_lag1", "dLON_lag1", "geo_speed_kmh", "geo_bearing",
    "geo_bear_sin", "geo_bear_cos", "curvature", "dlat_accel", "dlon_accel",
]


def build_params(quick: bool, seed: int = 42):
    n = 100 if quick else 300
    return {
        "RF_P": dict(n_estimators=n, n_jobs=2, random_state=seed),
        "XGB_P": dict(
            n_estimators=n, learning_rate=GLOBAL_LR, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            tree_method="hist", verbosity=0, n_jobs=2,
        ),
        "LGB_P": dict(
            n_estimators=n, learning_rate=GLOBAL_LR, max_depth=6,
            num_leaves=63, random_state=seed, verbose=-1, n_jobs=2,
        ),
        "CB_P": dict(
            iterations=n, learning_rate=GLOBAL_LR, depth=6,
            loss_function="MultiRMSE", random_seed=seed, verbose=0, thread_count=2,
        ),
        "STK_N": 100 if quick else 150,
        "SEED": seed,
    }


def subset_columns(feature_cols: list[str], sname: str) -> list[int]:
    if sname == "full":
        return list(range(len(feature_cols)))
    keys = SUBSET_KEYS[sname]
    return [i for i, c in enumerate(feature_cols) if any(k in c for k in keys)]


def make_tree_bases(p):
    return [
        ("CB", CatBoostRegressor(**p["CB_P"])),
        ("XGB", MultiOutputRegressor(xgb.XGBRegressor(**p["XGB_P"]))),
        ("LGB", MultiOutputRegressor(lgb.LGBMRegressor(**p["LGB_P"]))),
        ("RF", MultiOutputRegressor(RandomForestRegressor(**p["RF_P"]))),
    ]


def persistence_vectors(frame) -> np.ndarray:
    return np.column_stack([
        frame["dLAT_lag1"].fillna(0).values,
        frame["dLON_lag1"].fillna(0).values,
    ])


def nnls_predict(val_preds: list[np.ndarray], test_preds: list[np.ndarray], y_vl: np.ndarray) -> np.ndarray:
    if not val_preds:
        raise ValueError("no base predictions")
    X_vl_lat = np.hstack([p[:, 0:1] for p in val_preds])
    X_ts_lat = np.hstack([p[:, 0:1] for p in test_preds])
    w_lat, _ = nnls(X_vl_lat, y_vl[:, 0])
    w_lon, _ = nnls(np.hstack([p[:, 1:2] for p in val_preds]), y_vl[:, 1])
    pred_lat = X_ts_lat @ w_lat
    pred_lon = np.hstack([p[:, 1:2] for p in test_preds]) @ w_lon
    return np.column_stack([pred_lat, pred_lon])


def km_nnls_predict(
    val_preds: list[np.ndarray], test_preds: list[np.ndarray],
    y_vl: np.ndarray, persist_vl: np.ndarray, persist_ts: np.ndarray,
) -> np.ndarray:
    """NNLS in km space around persistence anchor."""
    y_vl_km = (y_vl - persist_vl) * DEG_TO_KM
    val_res = [(p - persist_vl) * DEG_TO_KM for p in val_preds]
    test_res = [(p - persist_ts) * DEG_TO_KM for p in test_preds]
    pred_km = nnls_predict(val_res, test_res, y_vl_km)
    return persist_ts + pred_km / DEG_TO_KM


def median_km(lat, lon, y_true, y_pred) -> float:
    err = track_errors_km(lat, lon, y_true, y_pred)
    return float(np.median(err)) if len(err) else np.nan


def train_stacking(X_tr, y_tr, X_vl, X_ts, stk_n: int, seed: int = 42, need_train_pred: bool = False):
    base_est = [
        ("rf", RandomForestRegressor(n_estimators=stk_n, n_jobs=2, random_state=seed)),
        ("xgb", xgb.XGBRegressor(n_estimators=stk_n, learning_rate=GLOBAL_LR, max_depth=6,
                                 random_state=seed, verbosity=0, n_jobs=2)),
        ("lgb", lgb.LGBMRegressor(n_estimators=stk_n, learning_rate=GLOBAL_LR,
                                  random_state=seed, verbose=-1, n_jobs=2)),
    ]
    sl = StackingRegressor(estimators=base_est, final_estimator=Ridge(), cv=3, n_jobs=1)
    sn = StackingRegressor(estimators=base_est, final_estimator=Ridge(), cv=3, n_jobs=1)
    sl.fit(X_tr, y_tr[:, 0])
    sn.fit(X_tr, y_tr[:, 1])
    stk_vl = np.column_stack([sl.predict(X_vl), sn.predict(X_vl)])
    stk_ts = np.column_stack([sl.predict(X_ts), sn.predict(X_ts)])
    stk_tr = np.column_stack([sl.predict(X_tr), sn.predict(X_tr)]) if need_train_pred else None
    return stk_vl, stk_ts, stk_tr


def train_prc(X_tr, persist_tr, y_tr, X_vl, persist_vl, X_ts, persist_ts, horizon: str, p):
    res_tr_km = (y_tr - persist_tr) * DEG_TO_KM
    if horizon == "48h":
        pl, pn = xgb.XGBRegressor(**p["XGB_P"]), xgb.XGBRegressor(**p["XGB_P"])
    else:
        pl, pn = lgb.LGBMRegressor(**p["LGB_P"]), lgb.LGBMRegressor(**p["LGB_P"])
    pl.fit(X_tr, res_tr_km[:, 0])
    pn.fit(X_tr, res_tr_km[:, 1])
    pred_vl = persist_vl + np.column_stack([pl.predict(X_vl), pn.predict(X_vl)]) / DEG_TO_KM
    pred_ts = persist_ts + np.column_stack([pl.predict(X_ts), pn.predict(X_ts)]) / DEG_TO_KM
    return pred_vl, pred_ts


def train_champion(name: str, X_tr, y_tr, X_vl, y_vl, X_ts, p):
    if name == "Random Forest":
        m = MultiOutputRegressor(RandomForestRegressor(**p["RF_P"]))
    elif name == "LightGBM":
        m = MultiOutputRegressor(lgb.LGBMRegressor(**p["LGB_P"]))
    elif name == "XGBoost":
        m = MultiOutputRegressor(xgb.XGBRegressor(**p["XGB_P"]))
    elif name == "CatBoost":
        m = CatBoostRegressor(**p["CB_P"])
    else:
        raise ValueError(name)
    if name == "CatBoost":
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
    else:
        m.fit(X_tr, y_tr)
    return m.predict(X_vl), m.predict(X_ts)


def train_subset_ensemble(
    subsets: list[str], feature_cols, X_tr, y_tr, X_vl, y_vl, X_ts, p,
) -> tuple[np.ndarray, np.ndarray]:
    val_parts, test_parts = [], []
    for sname in subsets:
        idx = subset_columns(feature_cols, sname)
        if len(idx) < 2:
            continue
        Xtr_s, Xvl_s, Xts_s = X_tr[:, idx], X_vl[:, idx], X_ts[:, idx]
        for alg_name, alg in make_tree_bases(p):
            if alg_name == "CB":
                alg.fit(Xtr_s, y_tr, eval_set=(Xvl_s, y_vl))
            else:
                alg.fit(Xtr_s, y_tr)
            val_parts.append(alg.predict(Xvl_s))
            test_parts.append(alg.predict(Xts_s))
    if not val_parts:
        raise ValueError("no subset models trained")
    sub_vl = nnls_predict(val_parts, val_parts, y_vl)
    sub_ts = nnls_predict(val_parts, test_parts, y_vl)
    return sub_vl, sub_ts


def train_cb_motionnn(
    X_tr, y_tr, X_vl, y_vl, X_ts,
    cb_vl, cb_ts, cb_tr, feature_cols, p,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.neural_network import MLPRegressor

    motion_idx = [feature_cols.index(c) for c in MOTION_FEAT_NAMES if c in feature_cols]

    def corr_features(X, cb_pred):
        cb_km = cb_pred * DEG_TO_KM
        motion = X[:, motion_idx] if motion_idx else np.zeros((len(X), 1))
        return np.hstack([cb_km, motion])

    Xc_tr = corr_features(X_tr, cb_tr)
    Xc_vl = corr_features(X_vl, cb_vl)
    Xc_ts = corr_features(X_ts, cb_ts)
    res_tr = (y_tr - cb_tr) * DEG_TO_KM
    res_vl = (y_vl - cb_vl) * DEG_TO_KM

    nn = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        learning_rate_init=GLOBAL_LR,
        max_iter=GLOBAL_EPOCHS_NN,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    nn.fit(Xc_tr, res_tr)
    corr_vl = nn.predict(Xc_vl) / DEG_TO_KM
    corr_ts = nn.predict(Xc_ts) / DEG_TO_KM
    return cb_vl + corr_vl, cb_ts + corr_ts


def get_horizon_frames(data, horizon: str, phase: int):
    """3h uses df_ts3 one-step frame in phase 2 (matches Table III baselines)."""
    use_ts3 = phase >= 2 and horizon == "3h"
    if use_ts3:
        y_tr = data["df_tr3"][["dLAT", "dLON"]].values
        y_vl = data["df_vl3"][["dLAT", "dLON"]].values
        y_ts = data["df_ts3"][["dLAT", "dLON"]].values
        lat_ts = data["df_ts3"]["LAT"].values
        lon_ts = data["df_ts3"]["LON"].values
        X_tr, X_vl, X_ts = data["X3_tr"], data["X3_vl"], data["X3_ts"]
        persist_tr = persistence_vectors(data["df_tr3"])
        persist_vl = persistence_vectors(data["df_vl3"])
        persist_ts = persistence_vectors(data["df_ts3"])
        tr_frame, vl_frame, ts_frame = data["df_tr3"], data["df_vl3"], data["df_ts3"]
    else:
        col = "dLAT_3h" if horizon == "3h" else f"dLAT_{horizon}"
        y_tr = data["mh_train"][[col, col.replace("dLAT", "dLON")]].values
        y_vl = data["mh_val"][[col, col.replace("dLAT", "dLON")]].values
        y_ts = data["mh_test"][[col, col.replace("dLAT", "dLON")]].values
        lat_ts = data["mh_test"]["LAT"].values
        lon_ts = data["mh_test"]["LON"].values
        X_tr, X_vl, X_ts = data["Xmh_tr"], data["Xmh_vl"], data["Xmh_ts"]
        persist_tr = persistence_vectors(data["mh_train"])
        persist_vl = persistence_vectors(data["mh_val"])
        persist_ts = persistence_vectors(data["mh_test"])
        tr_frame, vl_frame, ts_frame = data["mh_train"], data["mh_val"], data["mh_test"]
    return dict(
        y_tr=y_tr, y_vl=y_vl, y_ts=y_ts, lat_ts=lat_ts, lon_ts=lon_ts,
        X_tr=X_tr, X_vl=X_vl, X_ts=X_ts,
        persist_tr=persist_tr, persist_vl=persist_vl, persist_ts=persist_ts,
        tr_frame=tr_frame, vl_frame=vl_frame, ts_frame=ts_frame,
    )


def val_best_router(
    candidates_val: dict[str, np.ndarray],
    candidates_test: dict[str, np.ndarray],
    y_vl: np.ndarray,
    lat_vl: np.ndarray,
    lon_vl: np.ndarray,
) -> tuple[np.ndarray, str]:
    best_name = min(
        candidates_val,
        key=lambda k: median_km(lat_vl, lon_vl, y_vl, candidates_val[k]),
    )
    return candidates_test[best_name], best_name


def val_soft_router(
    candidates_val: dict[str, np.ndarray],
    candidates_test: dict[str, np.ndarray],
    y_vl: np.ndarray,
    lat_vl: np.ndarray,
    lon_vl: np.ndarray,
    top_k: int = 3,
) -> tuple[np.ndarray, str]:
    """Inverse-error weighted blend of top-k val candidates (more stable than hard argmin)."""
    scored = [
        (name, median_km(lat_vl, lon_vl, y_vl, pred))
        for name, pred in candidates_val.items()
    ]
    scored.sort(key=lambda x: x[1])
    top = scored[: min(top_k, len(scored))]
    weights = np.array([1.0 / (med + 1e-3) for _, med in top], dtype=float)
    weights /= weights.sum()
    blended = np.zeros_like(candidates_test[top[0][0]])
    for w, (name, _) in zip(weights, top):
        blended += w * candidates_test[name]
    label = "soft(" + "+".join(n for n, _ in top[:2]) + ")"
    return blended, label


def val_hybrid_router(
    candidates_val: dict[str, np.ndarray],
    candidates_test: dict[str, np.ndarray],
    y_vl: np.ndarray,
    lat_vl: np.ndarray,
    lon_vl: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Pick hard argmin vs soft-top3 by validation median — both chosen on val only."""
    hard_name = min(
        candidates_val,
        key=lambda k: median_km(lat_vl, lon_vl, y_vl, candidates_val[k]),
    )
    hard_val = candidates_val[hard_name]
    hard_test = candidates_test[hard_name]

    scored = sorted(
        ((n, median_km(lat_vl, lon_vl, y_vl, p)) for n, p in candidates_val.items()),
        key=lambda x: x[1],
    )
    top = scored[: min(3, len(scored))]
    weights = np.array([1.0 / (med + 1e-3) for _, med in top], dtype=float)
    weights /= weights.sum()
    soft_val = np.zeros_like(hard_val)
    soft_test = np.zeros_like(hard_test)
    for w, (name, _) in zip(weights, top):
        soft_val += w * candidates_val[name]
        soft_test += w * candidates_test[name]
    soft_name = "soft(" + "+".join(n for n, _ in top[:2]) + ")"

    hard_med = median_km(lat_vl, lon_vl, y_vl, hard_val)
    soft_med = median_km(lat_vl, lon_vl, y_vl, soft_val)
    if soft_med < hard_med:
        return soft_test, soft_name
    return hard_test, hard_name


def train_stacking_residual(
    X_tr, y_tr, stk_tr, X_vl, y_vl, stk_vl, X_ts, stk_ts, p,
    algo: str = "lgb",
) -> tuple[np.ndarray, np.ndarray]:
    """LGB or XGB correction on km residuals from stacking."""
    res_tr = (y_tr - stk_tr) * DEG_TO_KM
    if algo == "xgb":
        lat_model = xgb.XGBRegressor(**p["XGB_P"])
        lon_model = xgb.XGBRegressor(**p["XGB_P"])
    else:
        lat_model = lgb.LGBMRegressor(**p["LGB_P"])
        lon_model = lgb.LGBMRegressor(**p["LGB_P"])
    lat_model.fit(X_tr, res_tr[:, 0])
    lon_model.fit(X_tr, res_tr[:, 1])
    corr_vl = np.column_stack([lat_model.predict(X_vl), lon_model.predict(X_vl)]) / DEG_TO_KM
    corr_ts = np.column_stack([lat_model.predict(X_ts), lon_model.predict(X_ts)]) / DEG_TO_KM
    return stk_vl + corr_vl, stk_ts + corr_ts


def fuse_sece_v3(
    val_parts, test_parts, y_vl, persist_vl, persist_ts,
    lat_vl, lon_vl, horizon: str,
    champion_val: dict[str, np.ndarray],
    champion_test: dict[str, np.ndarray],
    X_vl, X_ts, X_tr, y_tr, stk_tr, p,
) -> tuple[np.ndarray, str]:
    candidates_val: dict[str, np.ndarray] = dict(champion_val)
    candidates_test: dict[str, np.ndarray] = dict(champion_test)

    full_val = nnls_predict(val_parts, val_parts, y_vl)
    full_test = nnls_predict(val_parts, test_parts, y_vl)
    candidates_val["SECE NNLS full"] = full_val
    candidates_test["SECE NNLS full"] = full_test

    champ_vl = list(champion_val.values())
    champ_ts = list(champion_test.values())
    ch_val = nnls_predict(champ_vl, champ_vl, y_vl)
    ch_test = nnls_predict(champ_vl, champ_ts, y_vl)
    candidates_val["SECE NNLS champions"] = ch_val
    candidates_test["SECE NNLS champions"] = ch_test

    if horizon in ("24h", "48h"):
        km_val = km_nnls_predict(champ_vl, champ_vl, y_vl, persist_vl, persist_vl)
        km_test = km_nnls_predict(champ_vl, champ_ts, y_vl, persist_vl, persist_ts)
        candidates_val["SECE km-NNLS"] = km_val
        candidates_test["SECE km-NNLS"] = km_test

    if horizon == "48h":
        lgb_vl = champion_val["LightGBM"]
        lgb_ts = champion_test["LightGBM"]
        prc_vl = champion_val["Persistence Residual Cascade"]
        prc_ts = champion_test["Persistence Residual Cascade"]
        lp_val = nnls_predict([lgb_vl, prc_vl], [lgb_vl, prc_vl], y_vl)
        lp_test = nnls_predict([lgb_vl, prc_vl], [lgb_ts, prc_ts], y_vl)
        candidates_val["LGB+PRC NNLS"] = lp_val
        candidates_test["LGB+PRC NNLS"] = lp_test
        lpkm_val = km_nnls_predict([lgb_vl, prc_vl], [lgb_vl, prc_vl], y_vl, persist_vl, persist_vl)
        lpkm_test = km_nnls_predict([lgb_vl, prc_vl], [lgb_ts, prc_ts], y_vl, persist_vl, persist_ts)
        candidates_val["LGB+PRC km-NNLS"] = lpkm_val
        candidates_test["LGB+PRC km-NNLS"] = lpkm_test

    if horizon == "3h":
        rf_vl, rf_ts = champion_val["Random Forest"], champion_test["Random Forest"]
        stk_vl, stk_ts = champion_val["Stacking Ensemble"], champion_test["Stacking Ensemble"]
        rs_val = nnls_predict([rf_vl, stk_vl], [rf_vl, stk_vl], y_vl)
        rs_test = nnls_predict([rf_vl, stk_vl], [rf_ts, stk_ts], y_vl)
        candidates_val["RF+STK NNLS"] = rs_val
        candidates_test["RF+STK NNLS"] = rs_test
        sr_vl, sr_ts = train_stacking_residual(
            X_tr, y_tr, stk_tr, X_vl, y_vl, stk_vl, X_ts, stk_ts, p,
        )
        candidates_val["STK+residual LGB"] = sr_vl
        candidates_test["STK+residual LGB"] = sr_ts

    return val_best_router(candidates_val, candidates_test, y_vl, lat_vl, lon_vl)


def fuse_sece_v4(
    val_parts, test_parts, y_vl, persist_vl, persist_ts,
    lat_vl, lon_vl, horizon: str,
    champion_val: dict[str, np.ndarray],
    champion_test: dict[str, np.ndarray],
    X_vl, X_ts, X_tr, y_tr, stk_tr, stk_vl, stk_ts, p,
) -> tuple[np.ndarray, str]:
    """Phase 4: all-horizon stacking residuals + km-NNLS + hybrid router."""
    candidates_val: dict[str, np.ndarray] = dict(champion_val)
    candidates_test: dict[str, np.ndarray] = dict(champion_test)

    full_val = nnls_predict(val_parts, val_parts, y_vl)
    full_test = nnls_predict(val_parts, test_parts, y_vl)
    candidates_val["SECE NNLS full"] = full_val
    candidates_test["SECE NNLS full"] = full_test

    champ_vl = list(champion_val.values())
    champ_ts = list(champion_test.values())
    ch_val = nnls_predict(champ_vl, champ_vl, y_vl)
    ch_test = nnls_predict(champ_vl, champ_ts, y_vl)
    candidates_val["SECE NNLS champions"] = ch_val
    candidates_test["SECE NNLS champions"] = ch_test

    # km-NNLS on all bases (12h+) and champions (24h+)
    km_full_val = km_nnls_predict(val_parts, val_parts, y_vl, persist_vl, persist_vl)
    km_full_test = km_nnls_predict(val_parts, test_parts, y_vl, persist_vl, persist_ts)
    candidates_val["SECE km-NNLS full"] = km_full_val
    candidates_test["SECE km-NNLS full"] = km_full_test

    km_val = km_nnls_predict(champ_vl, champ_vl, y_vl, persist_vl, persist_vl)
    km_test = km_nnls_predict(champ_vl, champ_ts, y_vl, persist_vl, persist_ts)
    candidates_val["SECE km-NNLS"] = km_val
    candidates_test["SECE km-NNLS"] = km_test

    # Stacking residual correction — all horizons (Phase 3 was 3h-only)
    sr_lgb_vl, sr_lgb_ts = train_stacking_residual(
        X_tr, y_tr, stk_tr, X_vl, y_vl, stk_vl, X_ts, stk_ts, p, algo="lgb",
    )
    candidates_val["STK+residual LGB"] = sr_lgb_vl
    candidates_test["STK+residual LGB"] = sr_lgb_ts

    sr_xgb_vl, sr_xgb_ts = train_stacking_residual(
        X_tr, y_tr, stk_tr, X_vl, y_vl, stk_vl, X_ts, stk_ts, p, algo="xgb",
    )
    candidates_val["STK+residual XGB"] = sr_xgb_vl
    candidates_test["STK+residual XGB"] = sr_xgb_ts

    rf_vl, rf_ts = champion_val["Random Forest"], champion_test["Random Forest"]
    rs_val = nnls_predict([rf_vl, stk_vl], [rf_vl, stk_vl], y_vl)
    rs_test = nnls_predict([rf_vl, stk_vl], [rf_ts, stk_ts], y_vl)
    candidates_val["RF+STK NNLS"] = rs_val
    candidates_test["RF+STK NNLS"] = rs_test

    prc_vl = champion_val["Persistence Residual Cascade"]
    prc_ts = champion_test["Persistence Residual Cascade"]
    sp_val = nnls_predict([stk_vl, prc_vl], [stk_vl, prc_vl], y_vl)
    sp_test = nnls_predict([stk_vl, prc_vl], [stk_ts, prc_ts], y_vl)
    candidates_val["STK+PRC NNLS"] = sp_val
    candidates_test["STK+PRC NNLS"] = sp_test

    if horizon in ("24h", "48h"):
        lgb_vl = champion_val["LightGBM"]
        lgb_ts = champion_test["LightGBM"]
        lp_val = nnls_predict([lgb_vl, prc_vl], [lgb_vl, prc_vl], y_vl)
        lp_test = nnls_predict([lgb_vl, prc_vl], [lgb_ts, prc_ts], y_vl)
        candidates_val["LGB+PRC NNLS"] = lp_val
        candidates_test["LGB+PRC NNLS"] = lp_test
        lpkm_val = km_nnls_predict([lgb_vl, prc_vl], [lgb_vl, prc_vl], y_vl, persist_vl, persist_vl)
        lpkm_test = km_nnls_predict([lgb_vl, prc_vl], [lgb_ts, prc_ts], y_vl, persist_vl, persist_ts)
        candidates_val["LGB+PRC km-NNLS"] = lpkm_val
        candidates_test["LGB+PRC km-NNLS"] = lpkm_test

    return val_hybrid_router(candidates_val, candidates_test, y_vl, lat_vl, lon_vl)


def fuse_sece_v2(
    val_parts, test_parts, y_vl, persist_vl, persist_ts,
    lat_vl, lon_vl, horizon: str,
    champion_val: dict[str, np.ndarray],
    champion_test: dict[str, np.ndarray],
) -> tuple[np.ndarray, str]:
    direct_val = nnls_predict(val_parts, val_parts, y_vl)
    direct_test = nnls_predict(val_parts, test_parts, y_vl)

    if horizon in ("48h", "24h"):
        km_val = km_nnls_predict(val_parts, val_parts, y_vl, persist_vl, persist_vl)
        km_test = km_nnls_predict(val_parts, test_parts, y_vl, persist_vl, persist_ts)
        if median_km(lat_vl, lon_vl, y_vl, km_val) <= median_km(lat_vl, lon_vl, y_vl, direct_val):
            direct_val, direct_test = km_val, km_test

    candidates_val = dict(champion_val)
    candidates_test = dict(champion_test)
    candidates_val["SECE NNLS"] = direct_val
    candidates_test["SECE NNLS"] = direct_test
    return val_best_router(candidates_val, candidates_test, y_vl, lat_vl, lon_vl)


def run_sece_v2(
    phase: int = 3,
    quick: bool = False,
    seed: int = 42,
    save_preds: bool = False,
    quiet: bool = False,
    export_all_test_preds_dir: Path | None = None,
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if save_preds:
        PREDS_DIR.mkdir(parents=True, exist_ok=True)
    p = build_params(quick, seed=seed)
    label = f"phase{phase}" + ("_quick" if quick else "")
    t0 = time.time()

    if not quiet:
        print(f"SECE v2 {label} (seed={seed}) — loading pipeline …", flush=True)
    data = load_pipeline_data(seed=seed)
    feature_cols = data["feature_cols"]

    if phase >= 4:
        sece_label = "SECE v2 Phase4"
    elif phase >= 3:
        sece_label = "SECE v2 Phase3"
    elif phase >= 2:
        sece_label = "SECE v2 Phase2"
    else:
        sece_label = "SECE v2"

    rank_rows = []

    for horizon in H_LABELS:
        if phase >= 3:
            subsets = SECE_SUBSETS_P3_3H if horizon == "3h" else SECE_SUBSETS_P3_OTHER
        elif phase >= 2:
            subsets = SECE_SUBSETS_P2
        else:
            subsets = SECE_SUBSETS_P1
        if not quiet:
            print(f"\n=== Horizon {horizon} (subsets={subsets}) ===", flush=True)
        fr = get_horizon_frames(data, horizon, phase)
        y_tr, y_vl, y_ts = fr["y_tr"], fr["y_vl"], fr["y_ts"]
        X_tr, X_vl, X_ts = fr["X_tr"], fr["X_vl"], fr["X_ts"]
        lat_ts, lon_ts = fr["lat_ts"], fr["lon_ts"]
        lat_vl = fr["vl_frame"]["LAT"].values
        lon_vl = fr["vl_frame"]["LON"].values
        ts_frame = fr["ts_frame"]
        persist_tr, persist_vl, persist_ts = fr["persist_tr"], fr["persist_vl"], fr["persist_ts"]

        horizon_preds = {}
        val_parts, test_parts = [], []
        champion_val, champion_test = {}, {}

        need_stk_tr = phase >= 4 or (phase >= 3 and horizon == "3h")
        stk_vl, stk_ts, stk_tr = train_stacking(
            X_tr, y_tr, X_vl, X_ts, p["STK_N"], seed=p["SEED"], need_train_pred=need_stk_tr,
        )
        if stk_tr is None:
            stk_tr = stk_vl
        prc_vl, prc_ts = train_prc(
            X_tr, persist_tr, y_tr, X_vl, persist_vl, X_ts, persist_ts, horizon, p,
        )
        horizon_preds["Stacking Ensemble"] = stk_ts
        horizon_preds["Persistence Residual Cascade"] = prc_ts
        champion_val["Stacking Ensemble"] = stk_vl
        champion_test["Stacking Ensemble"] = stk_ts
        champion_val["Persistence Residual Cascade"] = prc_vl
        champion_test["Persistence Residual Cascade"] = prc_ts
        val_parts.extend([stk_vl, prc_vl])
        test_parts.extend([stk_ts, prc_ts])

        cb_model = CatBoostRegressor(**p["CB_P"])
        cb_model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        cb_tr = cb_model.predict(X_tr)
        cb_vl = cb_model.predict(X_vl)
        cb_ts = cb_model.predict(X_ts)

        for champ in ["Random Forest", "LightGBM", "XGBoost"]:
            c_vl, c_ts = train_champion(champ, X_tr, y_tr, X_vl, y_vl, X_ts, p)
            horizon_preds[champ] = c_ts
            champion_val[champ] = c_vl
            champion_test[champ] = c_ts
            val_parts.append(c_vl)
            test_parts.append(c_ts)
        horizon_preds["CatBoost"] = cb_ts
        champion_val["CatBoost"] = cb_vl
        champion_test["CatBoost"] = cb_ts
        val_parts.append(cb_vl)
        test_parts.append(cb_ts)

        if phase >= 2:
            cbnn_vl, cbnn_ts = train_cb_motionnn(
                X_tr, y_tr, X_vl, y_vl, X_ts, cb_vl, cb_ts, cb_tr, feature_cols, p,
            )
            horizon_preds["CB+MotionNN"] = cbnn_ts
            champion_val["CB+MotionNN"] = cbnn_vl
            champion_test["CB+MotionNN"] = cbnn_ts
            val_parts.append(cbnn_vl)
            test_parts.append(cbnn_ts)

        sub_vl, sub_ts = train_subset_ensemble(
            subsets, feature_cols, X_tr, y_tr, X_vl, y_vl, X_ts, p,
        )
        val_parts.append(sub_vl)
        test_parts.append(sub_ts)

        if phase >= 4:
            sece_pred, router_pick = fuse_sece_v4(
                val_parts, test_parts, y_vl, persist_vl, persist_ts, lat_vl, lon_vl, horizon,
                champion_val, champion_test, X_vl, X_ts, X_tr, y_tr, stk_tr, stk_vl, stk_ts, p,
            )
            if not quiet:
                print(f"  [router selected: {router_pick}]", flush=True)
        elif phase >= 3:
            sece_pred, router_pick = fuse_sece_v3(
                val_parts, test_parts, y_vl, persist_vl, persist_ts, lat_vl, lon_vl, horizon,
                champion_val, champion_test, X_vl, X_ts, X_tr, y_tr, stk_tr, p,
            )
            if not quiet:
                print(f"  [router selected: {router_pick}]", flush=True)
        elif phase >= 2:
            sece_pred, router_pick = fuse_sece_v2(
                val_parts, test_parts, y_vl, persist_vl, persist_ts, lat_vl, lon_vl, horizon,
                champion_val, champion_test,
            )
            if not quiet:
                print(f"  [router selected: {router_pick}]", flush=True)
        else:
            sece_pred = nnls_predict(val_parts, test_parts, y_vl)
            router_pick = "NNLS"

        horizon_preds[sece_label] = sece_pred

        if export_all_test_preds_dir is not None:
            export_all_test_preds_dir.mkdir(parents=True, exist_ok=True)
            merged = ts_frame[["SID", "ISO_TIME", "LAT", "LON"]].copy()
            merged["ISO_TIME"] = pd.to_datetime(merged["ISO_TIME"])
            for name, pred in horizon_preds.items():
                merged[f"pred_dLAT_{name}"] = pred[:, 0]
                merged[f"pred_dLON_{name}"] = pred[:, 1]
            out_name = f"seed{seed}_predictions_{horizon}.csv"
            merged.to_csv(export_all_test_preds_dir / out_name, index=False)

        if save_preds:
            merged = ts_frame[["SID", "ISO_TIME", "LAT", "LON"]].copy()
            merged["ISO_TIME"] = pd.to_datetime(merged["ISO_TIME"])
            merged[f"pred_dLAT_{sece_label}"] = sece_pred[:, 0]
            merged[f"pred_dLON_{sece_label}"] = sece_pred[:, 1]
            if "CB+MotionNN" in horizon_preds:
                cbnn = horizon_preds["CB+MotionNN"]
                merged["pred_dLAT_CB+MotionNN"] = cbnn[:, 0]
                merged["pred_dLON_CB+MotionNN"] = cbnn[:, 1]
            merged.to_csv(PREDS_DIR / f"predictions_{horizon}_sece_v2.csv", index=False)

        meds = []
        for name, pred in horizon_preds.items():
            med = median_km(lat_ts, lon_ts, y_ts, pred)
            meds.append((name, med))
            if not quiet:
                print(f"  {name:35s}  Med={med:.3f} km", flush=True)
        meds.sort(key=lambda x: x[1])
        best_name, best_med = meds[0]
        for rank, (name, med) in enumerate(meds, start=1):
            rank_rows.append({
                "horizon": horizon, "model": name, "median_km": round(med, 3),
                "rank": rank, "gap_to_best": round(med - best_med, 3),
                "is_best": name == best_name, "phase": label, "seed": seed,
                "router_pick": router_pick if name == sece_label else "",
            })
        if not quiet:
            print(f"  >> Best: {best_name} ({best_med:.3f} km)", flush=True)
        gc.collect()

    rank_df = pd.DataFrame(rank_rows)
    rank_path = OUT_DIR / f"sece_v2_{label}_ranks.csv"
    if seed != 42:
        rank_path = OUT_DIR / f"sece_v2_{label}_seed{seed}_ranks.csv"
    rank_df.to_csv(rank_path, index=False)
    wins = rank_df[(rank_df["model"] == sece_label) & rank_df["is_best"]].shape[0]
    if not quiet:
        print(f"\n{sece_label} rank-1 horizons: {wins}/{len(H_LABELS)}", flush=True)
        print(f"Wrote {rank_path}", flush=True)
        if save_preds:
            print(f"Wrote predictions to {PREDS_DIR}", flush=True)
        print(f"Total time: {time.time() - t0:.1f}s", flush=True)
    return rank_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument("--quick", action="store_true", help="fewer trees / faster NN")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-preds", action="store_true", help="export key-safe test predictions")
    args = parser.parse_args()
    epochs = 50 if args.quick else 100
    globals()["GLOBAL_EPOCHS_NN"] = epochs
    sys.exit(0 if run_sece_v2(
        phase=args.phase, quick=args.quick, seed=args.seed, save_preds=args.save_preds,
    ) is not None else 1)
