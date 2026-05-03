from paper_engine import PaperEngine
import yfinance as yf
import pandas as pd
from alpha_engine import alpha_engine
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

engine = PaperEngine()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_market_data():
    tickers = [
    "AAPL","TSLA","NVDA","SPY",
    "MSFT","AMZN","META","GOOGL",
    "AMD","NFLX","PLTR","INTC",
    "COIN","NIO","RIVN","QQQ"
]

    data = {}

    for t in tickers:

        df = yf.download(t, period="5d", interval="5m")

        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

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

        ai_score = engine.predict_success(sig["signal"], price)

        trade = engine.execute(
            ticker,
            sig["signal"],
            price,
            sig.get("options", {})
        )
if trade:
    trade["ai_score"] = ai_score  # or confidence (depending on your variable name)
        results[ticker] = {
            **sig,
            "paper_trade": trade,
            "ai_trade_score": ai_score,
            "bias": engine.bias
        }

    engine.update(market_prices)

    return results


# 🚨 STEP 5 — MUST BE OUTSIDE scan()
@app.get("/stats")
def stats():

    trades = engine.trades

    completed = [t for t in trades if t.get("result") is not None]

    total = len(completed)
    wins = len([t for t in completed if t.get("result") == "WIN"])
    losses = len([t for t in completed if t.get("result") == "LOSS"])

    win_rate = wins / total if total > 0 else 0

    pnl_list = []

    for t in completed:
        if t.get("exit") is not None:
            if t["signal"] == "CALL":
                pnl = t["exit"] - t["entry"]
            else:
                pnl = t["entry"] - t["exit"]
            pnl_list.append(pnl)

    avg_pnl = sum(pnl_list) / len(pnl_list) if pnl_list else 0

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "bias": engine.bias,
        "avg_pnl": avg_pnl
    }