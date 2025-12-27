import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
import sys
from pathlib import Path

# Path Setup

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "rf_direction_model.pkl"
SRC_PATH = BASE_DIR / "src"
sys.path.append(str(SRC_PATH))

from src.features import Features
from src.inference_transformer import InferenceTransformer


# Load Model

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

feature_engineer = Features()
transformer = InferenceTransformer(model)


# UI

st.set_page_config(page_title="ML Stock Direction Indicator", page_icon="📈")
st.title("ML Stock Direction Indicator")
st.write(
    """
    This tool identifies **high-confidence bullish signals** using a machine learning model.
    It may abstain when confidence is low.
    """
)

st.markdown("""
### How to interpret this indicator

This is a **machine-learning based directional indicator**, not a trading bot.

- The model looks for **high-confidence bullish conditions**.
- It is intentionally **conservative** and may often return **NO SIGNAL**.
- A signal is shown **only when the model’s confidence exceeds the chosen threshold**.

**Important:**
- NO SIGNAL does *not* mean the market is bearish.
- It means current conditions are **ambiguous** based on historical patterns.
- This design prioritizes **precision over frequency**.
""")


# Input

ticker = st.text_input("Stock Ticker", value="AAPL")
lookback = st.slider("Lookback Window (days)", 60, 200, 120)
st.subheader("Signal Sensitivity")

threshold = st.slider(
    "Confidence Threshold",
    min_value=0.5,
    max_value=0.9,
    value=0.6,
    step=0.01,
    help="Higher threshold = fewer but more reliable signals"
)
st.info("0.6 is recommended value")

analyze = st.button("Analyze")

# Inference

if analyze:
    try:
        st.info("Fetching market data...")

        df = yf.download(
            ticker,
            period=f"{lookback}d",
            interval="1d",
            progress=False
        )

        if df.empty:
            st.error("No data found.")
            st.stop()

        df.reset_index(inplace=True)

        # Feature engineering
        df_feat = feature_engineer.add_features(df)

        # Latest candle
        latest = df_feat.iloc[-1:].select_dtypes(include=["number"])

        # Prediction
        df_feat = transformer.transform(df)
        latest = df_feat.iloc[-1:]
        prob_up = model.predict_proba(latest)[0][1]

        probs = model.predict_proba(df_feat)[:, 1]

        df_probs = df.iloc[-len(probs):].copy()
        df_probs["prob_up"] = probs

        st.subheader("Prediction")

        if prob_up >= threshold:
            st.success("Signal: **UP**")
            st.write(
            "The model detects a **high-confidence bullish pattern** based on historical data."
            )
        else:
            st.warning("Signal: **NO SIGNAL**")
            st.write(
            "The model does not find sufficient evidence to make a confident directional call."
            )

        st.metric("Confidence", f"{prob_up:.2%}")


        # Price chart
        st.subheader("Recent Price Action")
        st.line_chart(df.set_index("Date")["Close"].tail(60)) 

        st.subheader("Confidence History")
        st.line_chart(df_probs.set_index("Date")["prob_up"].tail(100))

        st.markdown("""
        ### Confidence history

        This chart shows how the model’s **confidence evolves over time**.

        - Values near **0.5** indicate uncertainty.
        - Spikes above the threshold indicate **rare, high-confidence conditions**.
        - Most days are intentionally filtered out.
        """)

        st.markdown("""
        ---
        **Disclaimer:**  
        This tool is for educational and research purposes only.  
        It does not constitute financial advice.
        """)


    except Exception as e:
        st.error(f"Error: {e}")