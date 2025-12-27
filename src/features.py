import pandas as pd
import numpy as np


class Features:
    def __init__(self):
        pass

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        price_cols = ["Open", "High", "Low", "Close", "Volume"]

        for col in price_cols:
            if col not in df.columns:
                raise ValueError(f"Required column missing: {col}")

            series = df.loc[:, col]

            # If multiple columns are returned, take the first one
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]

            # Force to Series explicitly
            series = pd.Series(series.values, index=df.index, name=col)

            df[col] = pd.to_numeric(series, errors="coerce")



        
        # Daily Returns
        df["return"] = df["Close"].pct_change()

        # Volatility
        df["volatility_20"] = df["return"].rolling(window=20).std()

        # Price Features
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
        df["co_diff"] = (df["Close"] - df["Open"]) / df["Open"]

        # MAs
        for window in [10, 20, 50]:
            df[f"sma_{window}"] = df["Close"].rolling(window).mean()
            df[f"ema_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()

        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        sma_20 = df["Close"].rolling(20).mean()
        std_20 = df["Close"].rolling(20).std()
        upper = sma_20 + 2 * std_20
        lower = sma_20 - 2 * std_20
        df["bb_width"] = (upper - lower) / sma_20

        # ATR
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # Volume
        df["vol_change"] = df["Volume"].pct_change()
        df["obv"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

        # clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df.dropna(inplace=True)

        return df
