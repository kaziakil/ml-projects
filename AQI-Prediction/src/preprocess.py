"""
preprocess.py – Data Preprocessing Pipeline

Reads the EDA-cleaned CSV, handles remaining missing values,
engineers features, derives the AQI target, and
produces train/test splits ready for model training.

Usage (standalone):
    python src/preprocess.py

Outputs:
    data/processed/train.csv
    data/processed/test.csv
    models/scaler.pkl
    models/feature_cols.pkl
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Constants
RAW_CLEANED_PATH = "data/processed/AirQualityUCI_cleaned.csv"
TRAIN_OUT_PATH = "data/processed/train.csv"
TEST_OUT_PATH = "data/processed/test.csv"
SCALER_OUT_PATH = "models/scaler.pkl"
FEATURE_OUT_PATH = "models/feature_cols.pkl"

TEST_FRACTION = 0.20

SENSOR_COLS = [
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
    "PT08.S4(NO2)", "PT08.S5(O3)",
]
GT_COLS  = ["CO(GT)", "NOx(GT)", "NO2(GT)"]
ENV_COLS = ["T", "RH", "AH", "C6H6(GT)"]

# GT columns are used ONLY for AQI derivation and lag engineering.
# They are excluded from the model feature set to prevent leakage.
GT_LEAKAGE_COLS = ["CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)"]

TARGET_COL = "AQI"


def load_data(path: str) -> pd.DataFrame:
    """Load the EDA-cleaned CSV with a DatetimeIndex."""
    log.info(f"Loading data from {path}")
    df = pd.read_csv(path, index_col="Datetime", parse_dates=True)
    df.sort_index(inplace=True)
    log.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-stage imputation:
      1. Time-based linear interpolation — honours the hourly cadence.
         Neighbouring hours are better predictors than the global mean.
      2. Forward-fill + back-fill for edge NaNs at series boundaries.

    Why not mean/median fill?
    Pollution is temporally autocorrelated. Hour 5's CO is better estimated
    from hours 4 and 6 than from a global mean computed across all seasons.
    Mean-fill destroys that temporal structure.

    Why not drop rows with missing values?
    The 366-row offline gap is contiguous. Dropping it would leave a silent
    discontinuity in the time series — lag features computed across that gap
    would be meaningless. Interpolation bridges it cleanly.
    """
    log.info("Imputing missing values via time-based interpolation")
    all_cols = SENSOR_COLS + GT_COLS + ENV_COLS

    df[all_cols] = (
        df[all_cols]
        .interpolate(method="time")   # linear between valid points in time
        .ffill()                       # fill leading NaN block if any
        .bfill()                       # fill trailing NaN block if any
    )

    remaining = df[all_cols].isnull().sum().sum()
    log.info(f"Missing values remaining after imputation: {remaining}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three layers of feature engineering:

    1. Cyclical temporal encoding
       Hour and month are circular — hour 23 is adjacent to hour 0.
       Sin/cos encoding preserves that topology. Raw integer encoding
       tells the model hour 23 and hour 0 are maximally far apart, which
       is wrong.

    2. Lag features
       Pollution at time t is strongly autocorrelated with t-1 (previous
       hour) and t-24 (same hour yesterday). These give the model memory
       of recent conditions without requiring a sequence architecture.
       All lags are on GT columns — shifted so they never include the
       current-hour value, preventing any form of leakage.

    3. Rolling statistics (24h window)
       Captures the local pollution baseline (mean) and volatility (std)
       over the past 24 hours. Shifted by 1 before rolling so the current
       hour is never included in its own rolling window.
    """
    log.info("Engineering features")

    # 1. Cyclical temporal encoding 
    hour  = df.index.hour
    month = df.index.month

    df["hour_sin"]   = np.sin(2 * np.pi * hour  / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * hour  / 24)
    df["month_sin"]  = np.sin(2 * np.pi * month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * month / 12)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # 2. Lag features 
    lag_cols = ["CO(GT)", "NOx(GT)", "NO2(GT)", "PT08.S1(CO)"]
    for col in lag_cols:
        df[f"{col}_lag1"]  = df[col].shift(1)   # 1-hour lag
        df[f"{col}_lag24"] = df[col].shift(24)  # same hour yesterday

    # 3. Rolling statistics (24h window, shifted by 1) 
    for col in ["CO(GT)", "NOx(GT)", "T"]:
        shifted = df[col].shift(1)
        df[f"{col}_roll24_mean"] = shifted.rolling(24).mean()
        df[f"{col}_roll24_std"]  = shifted.rolling(24).std()

    before = len(df)
    df.dropna(inplace=True)
    after  = len(df)
    log.info(f"Dropped {before - after} rows due to lag/rolling NaNs")
    log.info(f"Shape after feature engineering: {df.shape}")
    return df


def derive_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a composite AQI proxy from four ground-truth pollutants.

    Formula:
        AQI = 0.40 * CO_norm + 0.30 * NOx_norm + 0.20 * NO2_norm + 0.10 * C6H6_norm

    Weights reflect relative health impact — CO and NOx are primary
    traffic-sourced pollutants per WHO guidelines. C6H6 (benzene) is
    the weakest contributor at this concentration range.

    Normalisation: min-max per pollutant so each contributes on [0, 1].
    Global min/max computed on the full dataset before splitting —
    this is intentional. Min-max bounds define the scale of the problem.
    Using train-only bounds would mean test values could fall outside
    [0, 1], making the target unstable at inference time.

    Output range: [0, 1]
    """
    log.info("Deriving AQI target column")

    def _norm(s: pd.Series) -> pd.Series:
        return (s - s.min()) / (s.max() - s.min())

    df[TARGET_COL] = (
        0.40 * _norm(df["CO(GT)"])  +
        0.30 * _norm(df["NOx(GT)"]) +
        0.20 * _norm(df["NO2(GT)"]) +
        0.10 * _norm(df["C6H6(GT)"])
    )

    log.info(
        f"AQI — min: {df[TARGET_COL].min():.4f}  "
        f"max: {df[TARGET_COL].max():.4f}  "
        f"mean: {df[TARGET_COL].mean():.4f}"
    )
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return all columns except the target and raw GT pollutant columns.

    GT pollutant columns (CO(GT), NOx(GT), NO2(GT), C6H6(GT)) are excluded
    because they directly define AQI via the weighted sum above. Including
    them as features would let the model reconstruct the target algebraically
    — LinearRegression would score R²=1.0, which is leakage not performance.

    The model receives only:
      - Sensor readings (PT08.*)
      - Environmental variables (T, RH, AH)
      - Engineered temporal, lag, and rolling features
    This mirrors the realistic deployment scenario where ground-truth
    reference instruments are unavailable in real time.
    """
    exclude = set([TARGET_COL] + GT_LEAKAGE_COLS)
    return [c for c in df.columns if c not in exclude]


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split — never random on time-series data.

    Random splitting would leak future readings into the training set.
    For example, a row from January could end up in train while the
    previous hour (December) is in test — the model would effectively
    see the future during training, producing optimistic metrics that
    collapse at deployment.

    The last TEST_FRACTION of time is held out as test.
    """
    split_idx = int(len(df) * (1 - TEST_FRACTION))

    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    log.info(
        f"Train: {train.index.min().date()} → {train.index.max().date()} "
        f"({len(train)} rows)"
    )
    log.info(
        f"Test:  {test.index.min().date()} → {test.index.max().date()} "
        f"({len(test)} rows)"
    )
    return train, test


def scale_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Fit RobustScaler on train, transform both splits.

    Why RobustScaler over StandardScaler?
    Pollution data contains outliers — traffic spikes, industrial events,
    sensor anomalies. StandardScaler uses mean and std, both of which are
    sensitive to outliers. A single extreme spike compresses the scale for
    the majority of readings.
    RobustScaler uses median and IQR — outliers don't distort the scale.

    Why fit on train only?
    Fitting on the full dataset would let test statistics influence the
    scale seen during training — a subtle but real form of data leakage.
    The scaler must be saved and reused identically at inference time.
    """
    log.info("Scaling features with RobustScaler (fit on train only)")
    scaler = RobustScaler()

    train = train.copy()
    test  = test.copy()

    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols]  = scaler.transform(test[feature_cols])

    return train, test, scaler


def run_pipeline(
    raw_cleaned_path: str = RAW_CLEANED_PATH,
    train_out: str        = TRAIN_OUT_PATH,
    test_out: str         = TEST_OUT_PATH,
    scaler_out: str       = SCALER_OUT_PATH,
    feature_out: str      = FEATURE_OUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Full preprocessing pipeline. Orchestrates all steps in order.

    Returns (train_df, test_df, feature_cols) so train.py can import
    this function directly without re-reading files from disk.

    All intermediate steps log their output — if something breaks,
    the log tells you exactly which step failed and what the data
    looked like going in.
    """
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Execute pipeline steps in strict order
    df = load_data(raw_cleaned_path)
    df = impute_missing(df)
    df = engineer_features(df)
    df = derive_aqi(df)

    train, test  = split_data(df)
    feature_cols = get_feature_cols(train)

    train, test, scaler = scale_features(train, test, feature_cols)

    # Persist all artifacts
    train.to_csv(train_out)
    test.to_csv(test_out)
    joblib.dump(scaler,       scaler_out)
    joblib.dump(feature_cols, feature_out)

    log.info(f"Saved → {train_out}")
    log.info(f"Saved → {test_out}")
    log.info(f"Saved → {scaler_out}")
    log.info(f"Saved → {feature_out}")
    log.info(f"Feature count: {len(feature_cols)}")
    log.info(f"Features: {feature_cols}")

    return train, test, feature_cols


if __name__ == "__main__":
    train, test, feature_cols = run_pipeline()

    print(f"\n{'='*55}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*55}")
    print(f"Train shape    : {train.shape}")
    print(f"Test shape     : {test.shape}")
    print(f"Feature count  : {len(feature_cols)}")
    print(f"Target col     : {TARGET_COL}")
    print(f"\nFeatures:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:>2}. {col}")
    print(f"{'='*55}")

