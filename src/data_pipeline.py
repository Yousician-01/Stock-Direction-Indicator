import pandas as pd
from pathlib import Path

from data_loader import DataLoader
from features import Features
from label_generator import LabelGenerator


class DataPipeline:
    def __init__(
        self,
        raw_dir="data/raw",
        processed_dir="data/processed"
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.loader = DataLoader(raw_dir=self.raw_dir)
        self.feature_engineer = Features()
        self.label_generator = LabelGenerator()

    def run(
        self,
        ticker: str,
        start="2015-01-01",
        end=None,
        interval="1d"
    ) -> pd.DataFrame:
        """
        End-to-end data pipeline:
        raw -> features -> labels -> processed CSV
        """

        print(f"Downloading raw data for {ticker}")
        df = self.loader.download(
            ticker=ticker,
            start=start,
            end=end,
            interval=interval
        )

        print("Adding features")
        df = self.feature_engineer.add_features(df)

        print("Adding labels")
        df = self.label_generator.add_labels(df)

        # final cleanup
        df = df.dropna().reset_index(drop=True)

        out_file = self.processed_dir / f"{ticker.replace('.', '_')}_processed.csv"
        df.to_csv(out_file, index=False)

        print(f"Processed data saved to: {out_file}")
        print(f"Final shape: {df.shape}")

        return df

