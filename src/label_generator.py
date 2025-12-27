import pandas as pd


class LabelGenerator:
    def __init__(self):
        pass

    def add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Create target
        df["target"] = df["Close"].shift(-1) > df["Close"] - 1.5
        # Remove last row (no future price available)
        df = df.iloc[:-1]

        # Convert boolean → int
        df["target"] = df["target"].astype(int)

        return df
