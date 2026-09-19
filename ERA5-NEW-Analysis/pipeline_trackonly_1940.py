"""Track-only loader on the SAME 1940-2024 rows as pipeline_era5 (no ERA5 columns)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "Final_Cyclone_Pred_Results_P-3"
sys.path.insert(0, str(OLD))

import pipeline_common as pc  # noqa: E402
import pipeline_era5 as pe  # noqa: E402

# Same CSV / rows / QC / physics as ERA5 run — only feature columns differ.
DATA_PATH = pe.DATA_PATH
ALL_FEATURE_COLS = pc.ALL_FEATURE_COLS
H_LABELS = pc.H_LABELS
SPLIT_SEED = pc.SPLIT_SEED


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ALL_FEATURE_COLS if c in df.columns]


def load_pipeline_data(
    seed: int = SPLIT_SEED,
    train_storms: set | None = None,
    val_storms: set | None = None,
    test_storms: set | None = None,
):
    df_raw = pd.read_csv(DATA_PATH)
    df_qc = pc.apply_qc(df_raw)
    df = pc.apply_physics_features(df_qc)
    feature_cols = get_feature_cols(df)
    mh_df = pc.build_mh_df(df)
    if train_storms is None or val_storms is None or test_storms is None:
        train_storms, val_storms, test_storms = pc.storm_split(mh_df, seed=seed)

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
