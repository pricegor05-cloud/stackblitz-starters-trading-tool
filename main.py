from paper_engine import PaperEngine
import yfinance as yf
import pandas as pd
from alpha_engine import alpha_engine
from ai_engine import ai_predict
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

# =========================================================
# 🧠 GLOBAL CACHE + HOT STOCK SYSTEM
# =========================================================
stock_cache = {}
hot_stocks = set()

# -------------------------
# MARKET DATA
# -------------------------
def get_market_data():

    tickers = [
        "AAPL","TSLA","NVDA","SPY",
        "MSFT","AMZN","META","GOOGL",
        "AMD","NFLX","PLTR","INTC",
        "COIN","NIO","RIVN","QQQ"
    ]

    data = {}

    for t in tickers:

        try:
            df = yf.download(t, period="5d", interval="5m", progress=False)

            if df is None or df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna()

            if len(df) < 20:
                continue

            data[t] = {"df": df}

        except Exception:
            continue

    return data


# -------------------------
# MAIN SCAN ENDPOINT
# -------------------------
@app.get("/scan")
def scan():

    market = get_market_data()
    signals = alpha_engine(market, ai_predict)

    results = {}
    market_prices = {}

    for ticker, sig in signals.items():

        price = sig["price"]
        market_prices[ticker] = price

        # 🧠 HOT STOCK DETECTION (FIXED PLACEMENT)
        if sig.get("confidence", 0) > 0.78:
            hot_stocks.add(ticker)
            stock_cache[ticker] = sig

        # 🧠 AI SCORE
        ai_score = engine.predict_success(
            sig["signal"],
            price,
            sig.get("rsi", 50),
            sig.get("vwap", price)
        )

        # 🧠 EXECUTE TRADE
        trade = engine.execute(
            ticker,
            sig["signal"],
            price,
            sig.get("options", {})
        )

        if trade:
            trade["ai_score"] = ai_score

        results[ticker] = {
            **sig,
            "paper_trade": trade,
            "ai_trade_score": ai_score,
            "bias": engine.bias
        }

    engine.update(market_prices)

    return results


# -------------------------
# 🟡 HOT STOCKS ENDPOINT
# -------------------------
@app.get("/hot")
def hot():
    return list(hot_stocks)


# -------------------------
# 🟢 ON-DEMAND STOCK FETCH
# -------------------------
@app.get("/stock/{ticker}")
def stock(ticker: str):

    ticker = ticker.upper()

    if ticker in stock_cache:
        return stock_cache[ticker]

    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)

        if df is None or df.empty:
            return {"error": "no data"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()

        data = {"df": df}

        sig = alpha_engine(data, ai_predict)[ticker]

        stock_cache[ticker] = sig

        return sig

    except Exception:
        return {"error": "failed to load stock"}


# -------------------------
# STATS ENDPOINT
# -------------------------
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