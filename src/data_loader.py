import yfinance as yf
import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, raw_dir='data/raw'):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download(
            self,
            ticker: str,
            start: str = "2015-01-01",
            end: str | None=None,
            interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Download OHLCV data for a given stock ticker.
        """
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False
        )

        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        df.reset_index(inplace=True)

        # Save raw Data
        file_path = self.raw_dir / f"{ticker.replace('.', '_')}.csv"
        df.to_csv(file_path, index=False)

        return df