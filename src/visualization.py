from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_time_series(
    df,
    date_col: str = "date",
    target_col: str = "sales",
    title: str = "Time Series",
):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df[date_col], df[target_col], label=target_col)
    ax.set_title(title)
    ax.set_xlabel(date_col)
    ax.set_ylabel(target_col)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return ax


def plot_train_test_forecast(
    train_df,
    test_df,
    forecast_df,
    date_col: str = "date",
    target_col: str = "sales",
    forecast_col: str = "forecast",
    title: str = "Forecast",
):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_df[date_col], train_df[target_col], label="train")
    ax.plot(test_df[date_col], test_df[target_col], label="test")
    ax.plot(forecast_df[date_col], forecast_df[forecast_col], label="forecast")
    ax.set_title(title)
    ax.set_xlabel(date_col)
    ax.set_ylabel(target_col)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return ax


def save_plot(path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
