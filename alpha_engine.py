from options_engine import options_engine
from ai_engine import ai_predict
import pandas as pd
import numpy as np


# -------------------------
# STEP 3: TRADE FILTER
# -------------------------
def trade_filter(signal, confidence, regime):

    if signal == "HOLD":
        return False

    if regime == 0:  # high volatility
        return False

    if confidence < 0.62:
        return False

    return True


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


def safe_ai(df):
    try:
        return ai_predict(df)
    except:
        return {
            "ai_up_prob": 0.5,
            "ai_down_prob": 0.5,
            "confidence": 0.5
        }


# -------------------------
# STEP 2: REGIME DETECTOR
# -------------------------
def detect_regime(df):

    try:
        prices = df["Close"].values[-20:]

        returns = np.diff(prices)

        volatility = np.std(returns)
        trend = np.mean(returns)

        if volatility > 2:
            return 0  # unstable / chop

        if abs(trend) > 0.5:
            return 2  # trending

        return 1  # normal

    except:
        return 1


def alpha_engine(market_data):

    signals = {}

    for ticker, data in market_data.items():

        df = data.get("df")

        if df is None or df.empty or len(df) < 10:
            signals[ticker] = {
                "signal": "NO_DATA",
                "price": 0,
                "rsi": 50,
                "vwap": 0,
                "ai_up_prob": 0.5,
                "ai_down_prob": 0.5,
                "confidence": 0.5,
                "options": {"strategy": "none"}
            }
            continue

        df = df.dropna()

        # 🧠 AI LAYER
        ai = safe_ai(df)

        # 📊 INDICATORS
        rsi_series = rsi(df["Close"])
        vwap_series = vwap(df)

        price = last_value(df["Close"])
        rsi_val = last_value(rsi_series)
        vwap_val = last_value(vwap_series)

        try:
            momentum = float(df["Close"].iloc[-1] - df["Close"].iloc[-5])
        except:
            momentum = 0.0

        # ⚡ SCORE ENGINE
        score = 0

        score += 1 if price > vwap_val else -1

        if rsi_val < 35:
            score += 2
        elif rsi_val > 70:
            score -= 2

        score += 1 if momentum > 0 else -1

        # 📌 SIGNAL
        if score >= 2:
            signal = "CALL"
        elif score <= -2:
            signal = "PUT"
        else:
            signal = "HOLD"

        # -------------------------
        # 🔥 STEP 2: REGIME ADDED
        # -------------------------
        regime = detect_regime(df)

        # ⚡ OPTIONS ENGINE
        if signal in ["CALL", "PUT"]:
            try:
                option = options_engine(
                    ticker=ticker,
                    price=price,
                    signal=signal,
                    rsi=rsi_val,
                    vwap=vwap_val,
                    momentum=momentum
                )
            except:
                option = {"strategy": "fallback"}
        else:
            option = {"strategy": "none"}

        # 🧠 FINAL AI CONFIDENCE
        confidence = ai.get("confidence", 0.5)

        # -------------------------
        # 🚨 STEP 3: APPLY FILTER
        # -------------------------
        allowed = trade_filter(signal, confidence, regime)

        if not allowed:
            signal = "HOLD"
            option = {"strategy": "filtered"}

        # 📦 FINAL OUTPUT
        signals[ticker] = {
            "signal": signal,
            "score": score,
            "price": price,
            "rsi": rsi_val,
            "vwap": vwap_val,

            # 🧠 AI
            "ai_up_prob": ai.get("ai_up_prob", 0.5),
            "ai_down_prob": ai.get("ai_down_prob", 0.5),
            "confidence": confidence,

            # ⚡ CONTEXT (NEW)
            "regime": regime,

            # ⚡ OPTIONS
            "options": option
        }

    return signals