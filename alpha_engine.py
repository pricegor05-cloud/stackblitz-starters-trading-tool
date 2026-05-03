import numpy as np
import pandas as pd
from collections import defaultdict

# =========================================================
# 🧠 ADAPTIVE HEDGE FUND BRAIN v3
# =========================================================
class AIBrain:

    def __init__(self):
        self.global_memory = []
        self.ticker_memory = defaultdict(list)
        self.ticker_edge = defaultdict(float)

    def log(self, ticker, prediction, result):
        self.global_memory.append((prediction, result))
        self.ticker_memory[ticker].append((prediction, result))
        self._update_ticker_edge(ticker)

    def _update_ticker_edge(self, ticker):

        history = self.ticker_memory[ticker]

        if len(history) < 10:
            return

        wins = 0
        total = 0

        for p, r in history:
            if r is None:
                continue

            if (p > 0.5 and r == "WIN") or (p < 0.5 and r == "LOSS"):
                wins += 1

            total += 1

        if total > 0:
            self.ticker_edge[ticker] = (wins / total) - 0.5

    def accuracy(self):

        if len(self.global_memory) < 20:
            return 0.5

        correct = 0
        total = 0

        for p, r in self.global_memory:
            if r is None:
                continue

            if (p > 0.5 and r == "WIN") or (p < 0.5 and r == "LOSS"):
                correct += 1

            total += 1

        return correct / total if total else 0.5

    def ticker_bias(self, ticker):
        return self.ticker_edge.get(ticker, 0.0)


# =========================================================
# 🧠 GLOBAL INSTANCE
# =========================================================
ai_brain = AIBrain()


# =========================================================
# 🧠 SAFE AI WRAPPER
# =========================================================
def safe_ai(df, ticker="UNKNOWN", ai_predict=None):

    try:
        ai = ai_predict(df)

        if not isinstance(ai, dict):
            ai = {}

        up = float(ai.get("ai_up_prob", 0.5))
        down = float(ai.get("ai_down_prob", 0.5))
        conf = float(ai.get("confidence", 0.5))

        total = up + down
        if total > 0:
            up /= total
            down /= total

        global_acc = ai_brain.accuracy()
        ticker_bias = ai_brain.ticker_bias(ticker)

        conf = conf * (0.6 + global_acc + ticker_bias)
        conf = float(max(0.0, min(1.0, conf)))

        return {
            "ai_up_prob": up,
            "ai_down_prob": down,
            "confidence": conf
        }

    except Exception:
        return {
            "ai_up_prob": 0.5,
            "ai_down_prob": 0.5,
            "confidence": 0.5
        }


# =========================================================
# 📊 INDICATORS
# =========================================================
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
    return float(series.dropna().iloc[-1])


# =========================================================
# 🧠 MAIN ENGINE
# =========================================================
def alpha_engine(market_data, ai_predict=None):

    signals = {}

    for ticker, data in market_data.items():

        df = data.get("df")

        if df is None or df.empty or len(df) < 20:
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
        ai = safe_ai(df, ticker, ai_predict)

        # 📊 INDICATORS (SAFE)
        price = last_value(df["Close"])
        rsi_val = last_value(rsi(df["Close"]))
        vwap_val = last_value(vwap(df))

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

        # 📦 OUTPUT (ALWAYS SAFE)
        signals[ticker] = {
            "signal": signal,
            "score": score,
            "price": price,
            "rsi": rsi_val,
            "vwap": vwap_val,

            "ai_up_prob": ai["ai_up_prob"],
            "ai_down_prob": ai["ai_down_prob"],
            "confidence": ai["confidence"],

            "options": {"strategy": "basic"}
        }

    return signals