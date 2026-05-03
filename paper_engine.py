import json
import os
import numpy as np

DATA_FILE = "trades.json"


class PaperEngine:

    def __init__(self):
        self.trades = []
        self.bias = 0.0  # learning signal

        # load past trades
        self.load_trades()

    # -------------------------
    # STEP 1: EXECUTION + SAVE
    # -------------------------
    def execute(self, ticker, signal, price, options):

        if signal not in ["CALL", "PUT"]:
            return None

        trade = {
            "ticker": ticker,
            "signal": signal,
            "entry": float(price),
            "exit": None,
            "result": None,
            "options": options
        }

        self.trades.append(trade)
        self.save_trade(trade)

        return trade

    # -------------------------
    # STEP 2: SAVE TO DATASET
    # -------------------------
    def save_trade(self, trade):

        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(trade)

        with open(DATA_FILE, "w") as f:
            json.dump(data, f)

    def load_trades(self):

        if not os.path.exists(DATA_FILE):
            return

        with open(DATA_FILE, "r") as f:
            data = json.load(f)

        self.trades = data

        # rebuild bias from history
        wins = len([t for t in data if t.get("result") == "WIN"])
        losses = len([t for t in data if t.get("result") == "LOSS"])

        self.bias = (wins - losses) * 0.01

    # -------------------------
    # STEP 3: UPDATE + LEARNING
    # -------------------------
    def update(self, market_prices):

        for trade in self.trades:

            if trade["result"] is not None:
                continue

            price = market_prices.get(trade["ticker"])

            if price is None:
                continue

            price = float(price)

            # STOP LOSS
            if price <= trade["entry"] * 0.98:
                trade["exit"] = price
                trade["result"] = "LOSS"
                self.bias -= 0.01

            # TAKE PROFIT
            elif price >= trade["entry"] * 1.04:
                trade["exit"] = price
                trade["result"] = "WIN"
                self.bias += 0.01

        # re-save updated dataset
        with open(DATA_FILE, "w") as f:
            json.dump(self.trades, f)

    # -------------------------
    # 🧠 SIMPLE LEARNING MODEL
    # -------------------------
    def predict_success(self, signal, price):

        if len(self.trades) < 20:
            return 0.5

        wins = len([t for t in self.trades if t.get("result") == "WIN"])
        total = len([t for t in self.trades if t.get("result") is not None])

        win_rate = wins / total if total > 0 else 0.5

        bias_factor = self.bias

        signal_factor = 0.05 if signal == "CALL" else -0.05

        score = win_rate + bias_factor + signal_factor

        return float(max(0, min(1, score)))