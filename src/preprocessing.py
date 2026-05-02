from __future__ import annotations

from pathlib import Path

import pandas as pd


DATE_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "oil": "oil.csv",
    "holidays_events": "holidays_events.csv",
    "transactions": "transactions.csv",
}

NON_DATE_FILES = {
    "stores": "stores.csv",
    "sample_submission": "sample_submission.csv",
}


def load_raw_data(raw_path: str | Path = "data/raw") -> dict[str, pd.DataFrame]:
    raw_path = Path(raw_path)
    train_path = raw_path / "train.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            "Required file train.csv was not found. Download the Kaggle Store Sales "
            "competition files manually and place CSV files into data/raw/."
        )

    data: dict[str, pd.DataFrame] = {}

    for key, filename in DATE_FILES.items():
        file_path = raw_path / filename
        if file_path.exists():
            data[key] = pd.read_csv(file_path, parse_dates=["date"])

    for key, filename in NON_DATE_FILES.items():
        file_path = raw_path / filename
        if file_path.exists():
            data[key] = pd.read_csv(file_path)

    return data


def prepare_daily_series(
    train: pd.DataFrame,
    stores: pd.DataFrame | None = None,
    oil: pd.DataFrame | None = None,
    holidays: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    store_nbr: int = 1,
    family: str = "GROCERY I",
) -> pd.DataFrame:
    required_columns = {"date", "store_nbr", "family", "sales"}
    missing_columns = required_columns - set(train.columns)
    if missing_columns:
        raise ValueError(f"train.csv is missing required columns: {sorted(missing_columns)}")

    train = train.copy()
    train["date"] = pd.to_datetime(train["date"])

    store_data = train.loc[train["store_nbr"] == store_nbr].copy()
    if store_data.empty:
        raise ValueError(f"No records found for store_nbr={store_nbr}.")

    available_families = set(store_data["family"].dropna().unique())
    selected_family = family

    if family not in available_families:
        selected_family = (
            store_data.groupby("family", as_index=True)["sales"]
            .sum()
            .sort_values(ascending=False)
            .index[0]
        )

    series_data = store_data.loc[store_data["family"] == selected_family].copy()

    if "onpromotion" not in series_data.columns:
        series_data["onpromotion"] = 0

    daily = (
        series_data.groupby("date", as_index=False)
        .agg(sales=("sales", "sum"), onpromotion=("onpromotion", "sum"))
        .sort_values("date")
    )

    full_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_dates).rename_axis("date").reset_index()
    daily["sales"] = daily["sales"].fillna(0)
    daily["onpromotion"] = daily["onpromotion"].fillna(0)

    if oil is not None and {"date", "dcoilwtico"}.issubset(oil.columns):
        oil_daily = oil[["date", "dcoilwtico"]].copy()
        oil_daily["date"] = pd.to_datetime(oil_daily["date"])
        daily = daily.merge(oil_daily, on="date", how="left")
        daily["dcoilwtico"] = daily["dcoilwtico"].ffill().bfill()

    if transactions is not None and {"date", "store_nbr", "transactions"}.issubset(
        transactions.columns
    ):
        transactions_daily = transactions.loc[
            transactions["store_nbr"] == store_nbr, ["date", "transactions"]
        ].copy()
        transactions_daily["date"] = pd.to_datetime(transactions_daily["date"])
        transactions_daily = transactions_daily.groupby("date", as_index=False).agg(
            transactions=("transactions", "sum")
        )
        daily = daily.merge(transactions_daily, on="date", how="left")
        daily["transactions"] = daily["transactions"].fillna(0)

    if holidays is not None and "date" in holidays.columns:
        holidays_daily = holidays.copy()
        holidays_daily["date"] = pd.to_datetime(holidays_daily["date"])

        if "transferred" in holidays_daily.columns:
            holidays_daily = holidays_daily.loc[~holidays_daily["transferred"].fillna(False)]

        holidays_daily = holidays_daily.groupby("date", as_index=False).size()
        holidays_daily = holidays_daily.rename(columns={"size": "holidays_count"})
        daily = daily.merge(holidays_daily, on="date", how="left")
        daily["holidays_count"] = daily["holidays_count"].fillna(0).astype(int)
        daily["is_holiday"] = (daily["holidays_count"] > 0).astype(int)

    daily = daily.sort_values("date").reset_index(drop=True)
    daily.attrs["selected_store_nbr"] = store_nbr
    daily.attrs["selected_family"] = selected_family

    return daily


def handle_outliers_iqr(df: pd.DataFrame, target_col: str = "sales") -> pd.DataFrame:
    result = df.copy()

    q1 = result[target_col].quantile(0.25)
    q3 = result[target_col].quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        result[f"{target_col}_clean"] = result[target_col]
        return result

    lower_bound = max(0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    result[f"{target_col}_clean"] = result[target_col].clip(lower_bound, upper_bound)

    return result


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    result["year"] = result["date"].dt.year
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["week_of_year"] = result["date"].dt.isocalendar().week.astype(int)

    return result


def train_test_split_time_series(
    df: pd.DataFrame, test_days: int = 30
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= test_days:
        raise ValueError("DataFrame length must be greater than test_days.")

    sorted_df = df.sort_values("date").reset_index(drop=True)
    train_df = sorted_df.iloc[:-test_days].copy()
    test_df = sorted_df.iloc[-test_days:].copy()

    return train_df, test_df
