from sklearn.ensemble import RandomForestClassifier
import numpy as np
import json
import os

DATA_FILE = "trades.json"


class PaperEngine:

    def __init__(self):

        self.trades = []
        self.bias = 0.0

        # 🧠 ML MODEL
        self.model = RandomForestClassifier(n_estimators=150)
        self.model_trained = False

        self.load_trades()
        self.train_model()

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
            "options": options,

            # optional ML features (can be filled later)
            "rsi": options.get("rsi", 50),
            "vwap": options.get("vwap", price)
        }

        self.trades.append(trade)
        self.save_trade(trade)

        return trade

    # -------------------------
    # STEP 2: SAVE DATASET
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

        self.save_all()
        self.train_model()

    def save_all(self):
        with open(DATA_FILE, "w") as f:
            json.dump(self.trades, f)

    # -------------------------
    # 🧠 FEATURE ENGINEERING
    # -------------------------
    def build_features(self, trade):

        signal = 1 if trade["signal"] == "CALL" else 0

        entry = float(trade["entry"])

        rsi = trade.get("rsi", 50)
        vwap = trade.get("vwap", entry)

        vwap_dist = (entry - vwap) / entry

        bias = self.bias

        return [signal, entry, rsi, vwap_dist, bias]

    # -------------------------
    # 🧠 TRAIN MODEL
    # -------------------------
    def train_model(self):

        X = []
        y = []

        for t in self.trades:

            if t.get("result") not in ["WIN", "LOSS"]:
                continue

            X.append(self.build_features(t))
            y.append(1 if t["result"] == "WIN" else 0)

        if len(X) < 25:
            return

        self.model.fit(X, y)
        self.model_trained = True

    # -------------------------
    # 🧠 REAL ML PREDICTION
    # -------------------------
    def predict_success(self, signal, price, rsi=50, vwap=None):

        if vwap is None:
            vwap = price

        fake_trade = {
            "signal": signal,
            "entry": price,
            "rsi": rsi,
            "vwap": vwap
        }

        if not self.model_trained:
            return 0.5

        X = [self.build_features(fake_trade)]

        prob = self.model.predict_proba(X)[0][1]

        return float(prob)