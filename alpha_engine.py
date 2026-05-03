import numpy as np
from collections import defaultdict

# -------------------------
# 🧠 ADAPTIVE HEDGE FUND BRAIN
# -------------------------
class AIBrain:

    def __init__(self):
        # global memory
        self.global_memory = []

        # per-ticker intelligence
        self.ticker_memory = defaultdict(list)

        # learned weights per ticker
        self.ticker_edge = defaultdict(lambda: 0.0)

    # log trade result
    def log(self, ticker, prediction, result):
        self.global_memory.append((prediction, result))
        self.ticker_memory[ticker].append((prediction, result))

        self._update_ticker_edge(ticker)

    # update per-stock performance bias
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
            self.ticker_edge[ticker] = (wins / total) - 0.5  # centered bias

    # global accuracy
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

    # per ticker confidence boost
    def ticker_bias(self, ticker):
        return self.ticker_edge.get(ticker, 0.0)


# global brain instance
ai_brain = AIBrain()


# -------------------------
# 🧠 SAFE AI WRAPPER (HEDGE FUND VERSION)
# -------------------------
def safe_ai(df, ticker="UNKNOWN"):

    try:
        ai = ai_predict(df)

        if not isinstance(ai, dict):
            ai = {}

        up = float(ai.get("ai_up_prob", 0.5))
        down = float(ai.get("ai_down_prob", 0.5))
        conf = float(ai.get("confidence", 0.5))

        # normalize probabilities
        total = up + down
        if total > 0:
            up /= total
            down /= total

        # -------------------------
        # 🧠 HEDGE FUND CALIBRATION
        # -------------------------
        global_acc = ai_brain.accuracy()
        ticker_bias = ai_brain.ticker_bias(ticker)

        # adaptive scaling
        conf = conf * (0.6 + global_acc + ticker_bias)

        # clamp
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