import pandas as pd


def time_series_split(df: pd.DataFrame, train_ratio: float = 0.8):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    NON_FEATURES = ["Date", "target"]

    X_train = train_df.drop(columns=NON_FEATURES)
    y_train = train_df["target"]

    X_test  = test_df.drop(columns=NON_FEATURES)
    y_test  = test_df["target"]

    return X_train, y_train, X_test, y_test
