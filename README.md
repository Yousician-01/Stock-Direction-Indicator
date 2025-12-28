# ML Stock Direction Indicator

![CI](https://github.com/Yousician-01/Stock-Direction-Indicator/actions/workflows/ci.yml/badge.svg)
![Docker](https://img.shields.io/docker/pulls/atharvaraut01/ml-stock-indicator)

A machine-learning–based **stock direction indicator** that identifies **high-confidence bullish conditions** from historical price data.  
The system is intentionally conservative and abstains from making predictions when confidence is low.

> This project is designed as a **decision-support tool**, not a trading bot.

---

## Overview

Financial markets are noisy, especially at short time horizons.  
Instead of attempting to predict **every** price movement, this project focuses on **selective, high-confidence signals**.

The model:
- Prioritizes **precision over recall**
- Produces a signal only when confidence is sufficiently high
- Returns **NO SIGNAL** for most market conditions

This design favors **trust and interpretability** over misleading accuracy.

---

## Core Idea

Most ML market models fail because they:
- optimize for accuracy on noisy data
- force predictions in ambiguous conditions
- hide uncertainty from users

This project takes the opposite approach:

> **Silence is better than a wrong prediction.**

The indicator speaks **only when historical patterns strongly align**.

---

## System Architecture
```
User (Streamlit UI)
        ↓
Live Market Data (Yahoo Finance)
        ↓
Feature Engineering (technical indicators)
        ↓
Trained Random Forest Model
        ↓
Probability Output
        ↓
UP Signal or NO SIGNAL

```

---

## Machine Learning Approach

### Model
- **Random Forest Classifier**
- Chosen for:
  - robustness to noisy features
  - stability on tabular data
  - interpretability

### Features
Derived from OHLCV data, including:
- returns & log-returns
- volatility measures
- moving averages (SMA, EMA)
- momentum indicators (RSI, MACD)
- volatility bands
- volume-based features

### Labeling Strategy
The training label was intentionally **relaxed** to reduce noise:
``` UP = tomorrow’s close is not significantly lower than today’s close ```
This avoids micro-fluctuations and improves signal stability.

---

## Model Behavior & Evaluation

### Why accuracy is misleading
Daily stock direction is close to random.  
Accuracy penalizes conservative models and rewards noisy ones.

Instead, this project focuses on:
- **Precision** (when the model signals UP, how often it is correct)
- **Selectivity** (how often the model chooses to speak)
- **Confidence behavior over time**

### Final Model Characteristics
- High precision on bullish signals
- Low recall by design
- Abstains in ambiguous conditions
- Produces rare but interpretable signals

---

## Streamlit Application

The Streamlit app allows users to:
- Select a stock ticker
- Adjust the confidence threshold
- View:
  - current signal (UP / NO SIGNAL)
  - model confidence
  - historical confidence over time
  - recent price action

### Interpreting the Output

- **UP**  
  → The model detects a high-confidence bullish pattern.

- **NO SIGNAL**  
  → Current market conditions are ambiguous.

A confidence slider allows users to control how strict the indicator should be.

---

## Project Structure

```
project/
│
├── data/
│   ├── raw
│   └── processed
│ 
├── app.py                     # Streamlit app
├── requirements.txt
├── README.md
│
├── model/
│   └── rf_direction_model.pkl # Trained model artifact
│
├── src/
│   ├── __init__.py
│   ├── build_features.py
│   ├── data_loader.py 
│   ├── data_pipeline.py 
│   ├── features.py            # Feature engineering
│   ├── inference_transformer.py
│   ├── label_generator.py
│   ├── train_model.py
│   ├── train_test_split.py  
│   └── inference_transformer.py
│
└── .github/
    └── workflows/             # CI (added later)

```

---

## Getting Started

### Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Create a virtual environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the Streamlit app
```bash
streamlit run app.py
```

---

## Run with Docker (Recommended)
The application is fully containerized and available on Docker Hub.
```bash
docker pull atharvaraut01/ml-stock-indicator
docker run -p 8501:8501 atharvaraut01/ml-stock-indicator

```
Open: 
```http://localhost:8501```

---

## Limitations

This project intentionally does not:
- guarantee profitability
- model transaction costs
- optimize trading strategies
- predict every market movement

Known limitations:
- Daily price direction is highly noisy
- Performance varies across market regimes
- Designed as an indicator, not a trading system

---

## Disclaimer

This project is for educational and research purposes only.
It does not constitute financial advice.

---

## Engineering Roadmap
- Dockerized deployment
- CI with GitHub Actions
- Streamlit Cloud / container deployment
The ML model itself is frozen.

---

## Final Note

Markets are hard.
Silence is often the most honest output.

This project embraces that reality.