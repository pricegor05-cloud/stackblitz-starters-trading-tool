import numpy as np
import pandas as pd

# SAFE INDICATORS
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def vwap(df):
    return (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

def last(series):
    return float(series.dropna().iloc[-1])

# =========================
# MAIN ENGINE
# =========================
def alpha_engine(market_data, ai_predict=None):

    signals = {}

    for ticker, data in market_data.items():

        df = data.get("df")

        if df is None or df.empty or len(df) < 20:
            signals[ticker] = {
                "ticker": ticker,
                "signal": "NO_DATA",
                "price": 0,
                "rsi": 50,
                "vwap": 0,
                "confidence": 0.5,
                "ai_up_prob": 0.5,
                "ai_down_prob": 0.5,
                "score": 0,
                "hot": False,
                "options": {"strategy": "none"}
            }
            continue

        df = df.dropna()

        ai = ai_predict(df)

        price = last(df["Close"])
        rsi_val = last(rsi(df["Close"]))
        vwap_val = last(vwap(df))

        momentum = df["Close"].iloc[-1] - df["Close"].iloc[-5]

        score = 0
        score += 1 if price > vwap_val else -1
        score += 2 if rsi_val < 35 else -2 if rsi_val > 70 else 0
        score += 1 if momentum > 0 else -1

        if score >= 2:
            signal = "CALL"
        elif score <= -2:
            signal = "PUT"
        else:
            signal = "HOLD"

        signals[ticker] = {
            "ticker": ticker,
            "signal": signal,
            "price": price,
            "rsi": rsi_val,
            "vwap": vwap_val,
            "score": score,

            "ai_up_prob": ai["ai_up_prob"],
            "ai_down_prob": ai["ai_down_prob"],
            "confidence": ai["confidence"],

            "hot": ai["confidence"] > 0.75,

            "options": {"strategy": "basic"}
        }

    return signals