import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# Metrics for log and original space
# -----------------------------
def regression_report_log_and_orig(y_log_true, y_log_pred):
    y_true = np.exp(y_log_true)
    y_pred = np.exp(y_log_pred)

    # log-space
    mse_log = mean_squared_error(y_log_true, y_log_pred)
    rmse_log = np.sqrt(mse_log)
    mae_log = mean_absolute_error(y_log_true, y_log_pred)
    r2_log  = r2_score(y_log_true, y_log_pred)

    # original scale
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = (np.abs(y_pred - y_true) / y_true).mean() * 100

    return {
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


# -----------------------------
# Plots
# -----------------------------
def plot_history(history, title: str = "Training History"):
    plt.figure(figsize=(6,4))
    plt.plot(history.history.get("loss", []), label="Train")
    plt.plot(history.history.get("val_loss", []), label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def scatter_actual_vs_pred(y_true_log, y_pred_log, title: str):
    plt.figure(figsize=(6,4))
    plt.scatter(y_true_log, y_pred_log, alpha=0.6)
    lo = min(float(np.min(y_true_log)), float(np.min(y_pred_log)))
    hi = max(float(np.max(y_true_log)), float(np.max(y_pred_log)))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual log")
    plt.ylabel("Predicted log")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_timeseries(idx, actual, predicted, ylabel: str, title: str):
    ts_plot = pd.DataFrame({"date": idx, "actual": actual, "pred": predicted}).sort_values("date")
    plt.figure(figsize=(12,5))
    plt.plot(ts_plot["date"], ts_plot["actual"], label="Actual", alpha=0.9)
    plt.plot(ts_plot["date"], ts_plot["pred"], label="Predicted", alpha=0.9)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()