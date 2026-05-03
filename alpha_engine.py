from options_engine import options_engine
from ai_engine.py import ai_predict
import pandas as pd
import numpy as np


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def vwap(df):
    return (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()


def last_value(series):
    return float(series.squeeze().dropna().iloc[-1])


def alpha_engine(market_data):

    signals = {}

    for ticker, data in market_data.items():

        df = data["df"]

        if df is None or df.empty:
            signals[ticker] = {"signal": "NO_DATA"}
            continue

        df = df.dropna()

        # 🧠 AI LAYER
        ai = ai_predict(df)

        # 📊 INDICATORS
        rsi_series = rsi(df["Close"])
        vwap_series = vwap(df)

        price = last_value(df["Close"])
        rsi_val = last_value(rsi_series)
        vwap_val = last_value(vwap_series)

        momentum = float(df["Close"].iloc[-1] - df["Close"].iloc[-5])

        # ⚡ SCORE ENGINE
        score = 0

        if price > vwap_val:
            score += 1
        else:
            score -= 1

        if rsi_val < 35:
            score += 2
        elif rsi_val > 70:
            score -= 2

        if momentum > 0:
            score += 1
        else:
            score -= 1

        # 📌 SIGNAL
        if score >= 2:
            signal = "CALL"
        elif score <= -2:
            signal = "PUT"
        else:
            signal = "HOLD"

        # 🧠 OPTIONS ENGINE (STEP 3)
        option = options_engine(df, signal)

        # 📦 FINAL OUTPUT (STEP 4)
        signals[ticker] = {
            "signal": signal,
            "score": score,
            "price": price,
            "rsi": rsi_val,
            "vwap": vwap_val,

            # 🧠 AI LAYER
            "ai_up_prob": ai["ai_up_prob"],
            "ai_down_prob": ai["ai_down_prob"],
            "confidence": ai["confidence"],

            # ⚡ OPTIONS ENGINE OUTPUT
            "options": option
        }

    return signals