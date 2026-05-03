from paper_engine import PaperEngine

engine = PaperEngine()
import yfinance as yf
import pandas as pd
from alpha_engine import alpha_engine
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# allow StackBlitz
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_market_data():
    tickers = ["AAPL", "TSLA", "NVDA", "SPY"]

    data = {}

    for t in tickers:

        df = yf.download(t, period="5d", interval="5m")

        # safety copy
        df = df.copy()

        # flatten MultiIndex if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # remove bad rows
        df = df.dropna()

        if df is None or df.empty:
            continue

        data[t] = {"df": df}

    return data


@app.get("/scan")
def scan():

    market = get_market_data()
    signals = alpha_engine(market)

    results = {}

    market_prices = {}

    for ticker, sig in signals.items():

        price = sig["price"]
        market_prices[ticker] = price

        # 🧠 GET LEARNING PREDICTION (STEP 3 CONNECTED)
        ai_score = engine.predict_success(
            sig["signal"],
            price
        )

        # 🧠 PAPER TRADE EXECUTION
        trade = engine.execute(
            ticker,
            sig["signal"],
            price,
            sig.get("options", {})
        )

        results[ticker] = {
            **sig,

            # 💰 PAPER TRADING
            "paper_trade": trade,

            # 🧠 LEARNING OUTPUT
            "ai_trade_score": ai_score,

            # 📊 SYSTEM STATE
            "bias": engine.bias
        }

    # 🧠 STEP 4 — UPDATE LEARNING ENGINE
    engine.update(market_prices)

    return results