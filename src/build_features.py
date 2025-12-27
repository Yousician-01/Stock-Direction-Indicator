import sys
from pathlib import Path
import pandas as pd
from features import Features


PROJECT_ROOT = Path("..").resolve()
sys.path.append(str(PROJECT_ROOT))

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

ticker = "AAPL"

df = pd.read_csv(RAW_DIR / f"{ticker}.csv")

features = Features()
df_feat = features.add_features(df)

df_feat.to_csv(PROCESSED_DIR / f"{ticker}_features.csv", index=False)

print("Saved:", PROCESSED_DIR / f"{ticker}_features.csv")
