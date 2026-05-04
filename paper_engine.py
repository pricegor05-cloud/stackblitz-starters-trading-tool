import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

DATA_FILE = "trades.json"


class PaperEngine:

    def __init__(self):

        self.trades = []
        self.bias = 0.0

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )

        self.model_trained = False

        self.load_trades()
        self.train_model()

    # -------------------------
    # EXECUTE TRADE
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
            "rsi": options.get("rsi", 50),
            "vwap": options.get("vwap", price)
        }

        self.trades.append(trade)
        self.save_all()

        return trade

    # -------------------------
    # SAVE / LOAD
    # -------------------------
    def save_all(self):
        with open(DATA_FILE, "w") as f:
            json.dump(self.trades, f)

    def load_trades(self):

        if not os.path.exists(DATA_FILE):
            return

        with open(DATA_FILE, "r") as f:
            self.trades = json.load(f)

    # -------------------------
    # UPDATE MARKET
    # -------------------------
    def update(self, market_prices):

        for t in self.trades:

            if t["result"] is not None:
                continue

            price = market_prices.get(t["ticker"])
            if price is None:
                continue

            price = float(price)

            if price <= t["entry"] * 0.98:
                t["exit"] = price
                t["result"] = "LOSS"

            elif price >= t["entry"] * 1.04:
                t["exit"] = price
                t["result"] = "WIN"

        self.save_all()
        self.train_model()

    # -------------------------
    # FEATURES
    # -------------------------
    def build_features(self, t):

        signal = 1 if t["signal"] == "CALL" else 0
        entry = float(t["entry"])

        rsi = t.get("rsi", 50)
        vwap = t.get("vwap", entry)

        vwap_dist = (entry - vwap) / entry

        return [signal, entry, rsi, vwap_dist]

    # -------------------------
    # TRAIN MODEL
    # -------------------------
    def train_model(self):

        X, y = [], []

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
    # PREDICTION
    # -------------------------
    def predict_success(self, signal, price, rsi=50, vwap=None):

        if not self.model_trained:
            return 0.5

        if vwap is None:
            vwap = price

        fake = {
            "signal": signal,
            "entry": price,
            "rsi": rsi,
            "vwap": vwap
        }

        X = [self.build_features(fake)]

        return float(self.model.predict_proba(X)[0][1])