"""Shared cyclone pipeline utilities for audits, stats, and SECE v2."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datasets" / "bangladesh_nextstep_dataset_research_ready.csv"
PREDS_DIR = ROOT / "Total_Preds"

H_LABELS = ["3h", "12h", "24h", "48h"]
H_STEPS = [1, 4, 8, 16]
SPLIT_SEED = 42

ALL_FEATURE_COLS = [
    "LAT", "LON", "lat_lon_int",
    "LAT_lag1", "LON_lag1", "dLAT_lag1", "dLON_lag1",
    "LAT_lag2", "LON_lag2", "dLAT_lag2", "dLON_lag2",
    "LAT_lag3", "LON_lag3", "dLAT_lag3", "dLON_lag3",
    "geo_speed_kmh", "speed_sq", "geo_bearing",
    "geo_bear_sin", "geo_bear_cos",
    "DIST2LAND", "LANDFALL", "STORM_SPEED", "STORM_DIR", "NEWDELHI_WIND",
    "LAT_lag2_missing", "LON_lag2_missing", "dLAT_lag2_missing", "dLON_lag2_missing",
    "LAT_lag3_missing", "LON_lag3_missing", "dLAT_lag3_missing", "dLON_lag3_missing",
    "NEWDELHI_WIND_missing",
    "curvature", "bearing_stability", "speed_stability", "bearing_consistency",
    "season_sin", "season_cos", "dlat_accel", "dlon_accel", "dist_BoB",
    "dlat_trend", "dlon_trend", "bearing_momentum", "speed_trend",
    "recurvature_proxy", "dist2land_rate",
]

MODELS_3H_EXPORT_UNRELIABLE = frozenset(
    {
        "LSTM", "GRU", "BLSTM", "SLSTM", "CNN", "CNN-GRU", "CNN-LSTM",
        "GRU-LSTM", "Hybrid Transformer", "CB+MotionNN", "SECE Ensemble",
    }
)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def track_errors_km(lat, lon, y_true, y_pred) -> np.ndarray:
    mask = np.isfinite(y_true).all(axis=1) & np.isfinite(y_pred).all(axis=1)
    if not mask.any():
        return np.array([], dtype=float)
    return haversine_km(
        lat[mask] + y_true[mask, 0],
        lon[mask] + y_true[mask, 1],
        lat[mask] + y_pred[mask, 0],
        lon[mask] + y_pred[mask, 1],
    )


def apply_physics_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    doy = df["ISO_TIME"].dt.dayofyear.fillna(180)
    df["season_sin"] = np.sin(2 * np.pi * doy / 365)
    df["season_cos"] = np.cos(2 * np.pi * doy / 365)
    df["curvature"] = df.groupby("SID")["geo_bearing"].diff().fillna(0).abs()
    df["bearing_stability"] = 1.0 / (
        1.0
        + df.groupby("SID")["geo_bearing"].transform(
            lambda x: x.rolling(3, min_periods=1).std().fillna(0)
        )
    )
    df["speed_stability"] = 1.0 / (
        1.0
        + df.groupby("SID")["geo_speed_kmh"].transform(
            lambda x: x.rolling(3, min_periods=1).std().fillna(0)
        )
    )
    df["bearing_consistency"] = (
        df.groupby("SID")["geo_bear_sin"].transform(
            lambda x: x.rolling(3, min_periods=1).std().fillna(0)
        )
        + df.groupby("SID")["geo_bear_cos"].transform(
            lambda x: x.rolling(3, min_periods=1).std().fillna(0)
        )
    )
    df["dlat_accel"] = df["dLAT_lag1"] - df["dLAT_lag2"].fillna(df["dLAT_lag1"])
    df["dlon_accel"] = df["dLON_lag1"] - df["dLON_lag2"].fillna(df["dLON_lag1"])
    df["dist_BoB"] = haversine_km(df["LAT"].values, df["LON"].values, 15.0, 88.0)
    df["speed_sq"] = df["geo_speed_kmh"] ** 2
    df["lat_lon_int"] = df["LAT"] * df["LON"]
    df["dlat_trend"] = (
        df.groupby("SID")["dLAT_lag1"]
        .transform(
            lambda x: x.rolling(3, min_periods=1).apply(
                lambda v: (v.iloc[-1] - v.iloc[0]) if len(v) > 1 else 0, raw=False
            )
        )
        .fillna(0)
    )
    df["dlon_trend"] = (
        df.groupby("SID")["dLON_lag1"]
        .transform(
            lambda x: x.rolling(3, min_periods=1).apply(
                lambda v: (v.iloc[-1] - v.iloc[0]) if len(v) > 1 else 0, raw=False
            )
        )
        .fillna(0)
    )
    df["bearing_momentum"] = df.groupby("SID")["geo_bearing"].diff().fillna(0)
    df["speed_trend"] = (
        df.groupby("SID")["geo_speed_kmh"].transform(
            lambda x: x.rolling(3, min_periods=1).mean().fillna(0)
        )
        - df["geo_speed_kmh"].fillna(0)
    )
    df["recurvature_proxy"] = (df["LAT"] > 20).astype(float) * (
        df["geo_bear_cos"] > 0
    ).astype(float)
    df["dist2land_rate"] = df.groupby("SID")["DIST2LAND"].diff().fillna(0)
    return df


def apply_qc(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw[df_raw["flag_latnext_mismatch"] == 0].copy()
    df = df[df["flag_irregular_dt"] == 0]
    df = df.dropna(subset=["dLAT", "dLON", "LAT_next", "LON_next"])
    return df.reset_index(drop=True)


def build_mh_df(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for sid, grp in df.groupby("SID"):
        grp = grp.sort_values("ISO_TIME").reset_index(drop=True)
        n = len(grp)
        for t in range(n - max(H_STEPS)):
            row = grp.iloc[t].to_dict()
            valid = True
            for label, k in zip(H_LABELS, H_STEPS):
                fut = grp.iloc[t + k]
                row[f"LAT_{label}"] = fut["LAT"]
                row[f"LON_{label}"] = fut["LON"]
                row[f"dLAT_{label}"] = fut["LAT"] - grp.iloc[t]["LAT"]
                row[f"dLON_{label}"] = fut["LON"] - grp.iloc[t]["LON"]
                if pd.isna(row[f"LAT_{label}"]):
                    valid = False
            if valid:
                records.append(row)
    return pd.DataFrame(records)


def storm_split(mh_df: pd.DataFrame, seed: int = SPLIT_SEED):
    storms = np.array(mh_df["SID"].unique(), dtype=object)
    rng = np.random.RandomState(seed)
    rng.shuffle(storms)
    n = len(storms)
    n_tr = int(n * 0.70)
    n_vl = int(n * 0.15)
    train = set(storms[:n_tr])
    val = set(storms[n_tr : n_tr + n_vl])
    test = set(storms[n_tr + n_vl :])
    return train, val, test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ALL_FEATURE_COLS if c in df.columns]


def load_pipeline_data(
    seed: int = SPLIT_SEED,
    train_storms: set | None = None,
    val_storms: set | None = None,
    test_storms: set | None = None,
):
    df_raw = pd.read_csv(DATA_PATH)
    df_qc = apply_qc(df_raw)
    df = apply_physics_features(df_qc)
    feature_cols = get_feature_cols(df)
    mh_df = build_mh_df(df)
    if train_storms is None or val_storms is None or test_storms is None:
        train_storms, val_storms, test_storms = storm_split(mh_df, seed=seed)

    mh_train = mh_df[mh_df["SID"].isin(train_storms)].copy()
    mh_val = mh_df[mh_df["SID"].isin(val_storms)].copy()
    mh_test = mh_df[mh_df["SID"].isin(test_storms)].copy()
    df_tr3 = df[df["SID"].isin(train_storms)].copy()
    df_vl3 = df[df["SID"].isin(val_storms)].copy()
    df_ts3 = df[df["SID"].isin(test_storms)].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X3_tr = scaler.fit_transform(imputer.fit_transform(df_tr3[feature_cols].values))
    X3_vl = scaler.transform(imputer.transform(df_vl3[feature_cols].values))
    X3_ts = scaler.transform(imputer.transform(df_ts3[feature_cols].values))
    Xmh_tr = scaler.transform(imputer.transform(mh_train[feature_cols].values))
    Xmh_vl = scaler.transform(imputer.transform(mh_val[feature_cols].values))
    Xmh_ts = scaler.transform(imputer.transform(mh_test[feature_cols].values))

    return {
        "df": df,
        "mh_df": mh_df,
        "mh_train": mh_train,
        "mh_val": mh_val,
        "mh_test": mh_test,
        "df_tr3": df_tr3,
        "df_vl3": df_vl3,
        "df_ts3": df_ts3,
        "train_storms": train_storms,
        "val_storms": val_storms,
        "test_storms": test_storms,
        "feature_cols": feature_cols,
        "imputer": imputer,
        "scaler": scaler,
        "X3_tr": X3_tr,
        "X3_vl": X3_vl,
        "X3_ts": X3_ts,
        "Xmh_tr": Xmh_tr,
        "Xmh_vl": Xmh_vl,
        "Xmh_ts": Xmh_ts,
    }


def _prepare_eval_frame(base: pd.DataFrame, pred: pd.DataFrame, merge_keys: list[str]) -> pd.DataFrame:
    left = base.copy()
    left["ISO_TIME"] = pd.to_datetime(left["ISO_TIME"])
    pred = pred.copy()
    pred["ISO_TIME"] = pd.to_datetime(pred["ISO_TIME"])
    pred_cols = [c for c in pred.columns if c.startswith("pred_dLAT_") or c.startswith("pred_dLON_")]
    pred_only = pred[merge_keys + pred_cols].drop_duplicates(subset=merge_keys)
    return left.merge(pred_only, on=merge_keys, how="left")


def extract_model_names(columns) -> list[str]:
    models = []
    for col in columns:
        m = re.match(r"pred_dLAT_(.+)$", col)
        if m:
            models.append(m.group(1))
    return models


def load_eval_frames(data: dict) -> dict[str, pd.DataFrame]:
    merge_keys = ["SID", "ISO_TIME", "LAT", "LON"]
    pred3 = pd.read_csv(PREDS_DIR / "predictions_3h_all_models.csv")
    frames = {
        "3h": _prepare_eval_frame(data["df_ts3"], pred3, merge_keys),
    }
    for h in ["12h", "24h", "48h"]:
        pred = pd.read_csv(PREDS_DIR / f"predictions_{h}_all_models.csv")
        frames[h] = _prepare_eval_frame(data["mh_test"], pred, merge_keys)
    return frames


def save_test_predictions(
    ts_frame: pd.DataFrame,
    pred: np.ndarray,
    model_name: str,
    horizon: str,
    out_dir: Path,
) -> Path:
    """Save key-safe test predictions for stats overlay."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["SID", "ISO_TIME", "LAT", "LON"]
    out = ts_frame[keys].copy()
    out["ISO_TIME"] = pd.to_datetime(out["ISO_TIME"])
    safe_name = model_name.replace("/", "-")
    out[f"pred_dLAT_{safe_name}"] = pred[:, 0]
    out[f"pred_dLON_{safe_name}"] = pred[:, 1]
    path = out_dir / f"predictions_{horizon}_sece_v2.csv"
    out.to_csv(path, index=False)
    return path


def overlay_sece_v2_predictions(
    frames: dict[str, pd.DataFrame],
    preds_dir: Path,
    model_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Merge exported SECE v2 (or other) predictions into eval frames."""
    if model_names is None:
        model_names = ["SECE v2 Phase3", "CB+MotionNN"]
    merge_keys = ["SID", "ISO_TIME", "LAT", "LON"]
    updated = {}
    for horizon, frame in frames.items():
        f = frame.copy()
        f["ISO_TIME"] = pd.to_datetime(f["ISO_TIME"])
        pred_path = preds_dir / f"predictions_{horizon}_sece_v2.csv"
        if not pred_path.exists():
            updated[horizon] = f
            continue
        pred = pd.read_csv(pred_path)
        pred["ISO_TIME"] = pd.to_datetime(pred["ISO_TIME"])
        for model_name in model_names:
            safe_name = model_name.replace("/", "-")
            lat_col, lon_col = f"pred_dLAT_{safe_name}", f"pred_dLON_{safe_name}"
            if lat_col not in pred.columns:
                lat_col = f"pred_dLAT_{model_name}"
                lon_col = f"pred_dLON_{model_name}"
            if lat_col not in pred.columns:
                continue
            pred_only = pred[merge_keys + [lat_col, lon_col]].drop_duplicates(subset=merge_keys)
            pred_only = pred_only.rename(
                columns={lat_col: f"pred_dLAT_{model_name}", lon_col: f"pred_dLON_{model_name}"}
            )
            drop_cols = [c for c in f.columns if c in (f"pred_dLAT_{model_name}", f"pred_dLON_{model_name}")]
            f = f.drop(columns=drop_cols, errors="ignore")
            f = f.merge(pred_only, on=merge_keys, how="left")
        updated[horizon] = f
    return updated


def group_kfold_storm_splits(
    mh_df: pd.DataFrame,
    n_splits: int = 5,
    val_frac: float = 0.15,
    seed: int = SPLIT_SEED,
):
    """Yield (train_storms, val_storms, test_storms) for GroupKFold by SID."""
    from sklearn.model_selection import GroupKFold

    storms = np.array(sorted(mh_df["SID"].unique(), key=str))
    storm_idx = np.arange(len(storms))
    gkf = GroupKFold(n_splits=n_splits)
    rng = np.random.RandomState(seed)
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(storm_idx, groups=storm_idx)):
        test_storms = set(storms[te_idx])
        train_val = list(storms[tr_idx])
        rng.shuffle(train_val)
        n_val = max(1, int(len(train_val) * val_frac))
        val_storms = set(train_val[:n_val])
        train_storms = set(train_val[n_val:])
        yield fold, train_storms, val_storms, test_storms


def horizon_targets(frame: pd.DataFrame, horizon: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if horizon == "3h":
        true_dlat, true_dlon = "dLAT", "dLON"
    else:
        true_dlat, true_dlon = f"dLAT_{horizon}", f"dLON_{horizon}"
    lat = frame["LAT"].values
    lon = frame["LON"].values
    y_true = frame[[true_dlat, true_dlon]].values
    return lat, lon, y_true


def sample_errors_for_model(frame: pd.DataFrame, horizon: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    lat, lon, y_true = horizon_targets(frame, horizon)
    y_pred = frame[[f"pred_dLAT_{model}", f"pred_dLON_{model}"]].values
    mask = np.isfinite(y_true).all(axis=1) & np.isfinite(y_pred).all(axis=1)
    errors = track_errors_km(lat, lon, y_true, y_pred)
    sids = frame.loc[mask, "SID"].values
    return errors, sids


def per_storm_medians(errors: np.ndarray, sids: np.ndarray) -> dict[str, float]:
    if len(errors) == 0:
        return {}
    df = pd.DataFrame({"SID": sids, "error_km": errors})
    return df.groupby("SID")["error_km"].median().to_dict()
