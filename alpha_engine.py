import numpy as np
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
        conf = max(0.0, min(1.0, conf))

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
# 🚨 THIS IS THE LINE YOU WERE ASKING FOR (THE FIX)
# =========================================================
# YOU MUST CALL IT LIKE THIS INSIDE alpha_engine:

# ai = safe_ai(df, ticker, ai_predict)