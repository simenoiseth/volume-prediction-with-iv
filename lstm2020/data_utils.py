import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional

# -----------------------------
# General helpers
# -----------------------------
def set_seed(seed: int = 42):
    import tensorflow as tf
    np.random.seed(seed)
    tf.random.set_seed(seed)


def read_and_merge(train_path: str, test_path: str, date_col: str = "date") -> Tuple[pd.DataFrame, pd.Timestamp]:
    train = pd.read_csv(train_path, parse_dates=[date_col]).sort_values(date_col).reset_index(drop=True)
    test  = pd.read_csv(test_path,  parse_dates=[date_col]).sort_values(date_col).reset_index(drop=True)
    split_date = test[date_col].min()
    df = pd.concat([train, test], ignore_index=True).sort_values(date_col).reset_index(drop=True)
    return df, split_date


# -----------------------------
# Feature engineering for 5/21-day tasks
# -----------------------------
def build_features_5_21(df: pd.DataFrame,
                        use_vix: bool,
                        target: str) -> pd.DataFrame:
    """
    target: "target_5d" or "target_21d"
    use_vix: whether to add VIX-derived features.
    """
    df = df.copy()
    # Volume feature
    df["log_vol"] = np.log(df["sh_volume"])  # single base feature
    # Target on log scale
    if target not in df.columns:
        raise ValueError(f"`{target}` not found in dataframe.")
    df["y_log"] = np.log(df[target])

    if use_vix:
        df["vix_lag1"]   = df["vix_close"].shift(1)
        df["vix_change"] = df["vix_close"] - df["vix_lag1"]
        df["vix_5d_ma"]  = df["vix_close"].rolling(window=5).mean()
        needed = ["log_vol", "vix_close", "vix_lag1", "vix_change", "vix_5d_ma", "y_log"]
    else:
        needed = ["log_vol", "y_log"]

    df = df.dropna(subset=needed).reset_index(drop=True)
    return df


# -----------------------------
# Feature engineering for next-day task
# -----------------------------
def build_features_nextday(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_volume_t"] = np.log(df["sh_volume"])  # base signal at t

    # VIX features
    df["vix_lag1"]   = df["vix_close"].shift(1)
    df["vix_change"] = df["vix_close"] - df["vix_lag1"]
    df["vix_5d_ma"]  = df["vix_close"].rolling(window=5).mean()

    df.head()

    # Target
    if "target_volume" not in df.columns:
        raise ValueError("`target_volume` not found in dataframe for next-day task.")
    df["y_log"] = np.log(df["target_volume"])  # model in log space

    needed = ["log_volume_t", "vix_close", "vix_lag1", "vix_change", "vix_5d_ma", "y_log"]
    df = df.dropna(subset=needed).reset_index(drop=True)
    return df


# -----------------------------
# Train/test split and scaling
# -----------------------------
def split_by_date(df: pd.DataFrame, split_date: pd.Timestamp, date_col: str = "date"):
    start_date = pd.Timestamp("2020-01-01")
    df = df[df["date"] >= start_date].reset_index(drop=True)
    
    train_df = df[df[date_col] < split_date].copy()
    test_df  = df[df[date_col] >= split_date].copy()
    return train_df, test_df


def fit_transform_scalers(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]):
    X_scaler = StandardScaler().fit(train_df[feature_cols])
    train_scaled = train_df.copy()
    test_scaled  = test_df.copy()
    train_scaled[feature_cols] = X_scaler.transform(train_df[feature_cols])
    test_scaled[feature_cols]  = X_scaler.transform(test_df[feature_cols])
    return train_scaled, test_scaled, X_scaler


# -----------------------------
# Sequence builders
# -----------------------------
def make_sequences_generic(frame: pd.DataFrame,
                           window: int,
                           feature_cols: List[str],
                           target_col: str = "y_log",
                           date_col: str = "date"):
    X_list, y_list, idx_list = [], [], []
    values = frame[feature_cols].values.astype(np.float32)
    targets = frame[target_col].values.astype(np.float32)
    dates   = pd.to_datetime(frame[date_col])
    for t in range(window - 1, len(frame)):
        X_list.append(values[t - window + 1 : t + 1])
        y_list.append(targets[t])
        idx_list.append(dates.iloc[t])
    X = np.array(X_list)
    y = np.array(y_list)
    idx = pd.to_datetime(idx_list)
    return X, y, idx


def time_aware_train_val_split(X: np.ndarray, y: np.ndarray, val_frac: float = 0.1):
    val_size = max(int(len(X) * val_frac), 1)
    X_tr, X_val = X[:-val_size], X[-val_size:]
    y_tr, y_val = y[:-val_size], y[-val_size:]
    return X_tr, y_tr, X_val, y_val