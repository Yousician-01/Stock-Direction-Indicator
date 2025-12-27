import pandas as pd
from src.features import Features

class InferenceTransformer:
    def __init__(self, model):
        self.model = model
        self.feature_engineer = Features()
        self.n_features = model.n_features_in_

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Recreates the exact feature matrix used during training.
        """

        df = raw_df.copy()

        # Ensure Date column exists
        if "Date" not in df.columns:
            df.reset_index(inplace=True)

        # Apply SAME feature engineering
        df = self.feature_engineer.add_features(df)

        # Drop NaNs exactly like training
        df = df.dropna()

        # Keep only numeric columns
        X = df.select_dtypes(include=["number"])

        # Drop target if present (training did this)
        if "target" in X.columns:
            X = X.drop(columns=["target"])

        # Ensure correct feature count
        if X.shape[1] < self.n_features:
            raise ValueError(
                f"Not enough features after processing. "
                f"Expected {self.n_features}, got {X.shape[1]}"
            )

        # 🔥 CRITICAL: match training-time shape by position
        X = X.iloc[:, :self.n_features]

        return X
