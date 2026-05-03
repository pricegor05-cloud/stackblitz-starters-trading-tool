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
# 🧠 CACHE SYSTEM
# =========================================================
stock_cache = {}
hot_stocks = set()

# =========================================================
# MARKET DATA
# =========================================================
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

# =========================================================
# SCAN ENDPOINT
# =========================================================
@app.get("/scan")
def scan():

    market = get_market_data()
    signals = alpha_engine(market, ai_predict)

    results = {}
    market_prices = {}

    for ticker, sig in signals.items():

        if not isinstance(sig, dict):
            continue

        price = float(sig.get("price") or 0)
        confidence = float(sig.get("confidence") or 0.5)

        market_prices[ticker] = price

        # 🟡 HOT STOCK LOGIC
        if confidence > 0.78:
            hot_stocks.add(ticker)
            stock_cache[ticker] = sig

        # 🧠 AI SCORE
        ai_score = engine.predict_success(
            sig.get("signal", "HOLD"),
            price,
            sig.get("rsi", 50),
            sig.get("vwap", price)
        )

        # 💰 PAPER TRADE
        trade = engine.execute(
            ticker,
            sig.get("signal", "HOLD"),
            price,
            sig.get("options", {})
        )

        if trade:
            trade["ai_score"] = ai_score

        # 🧠 SAFE OUTPUT
        results[ticker] = {
            "ticker": ticker,
            "signal": sig.get("signal", "HOLD"),
            "price": price,

            "confidence": confidence,
            "ai_up_prob": float(sig.get("ai_up_prob") or 0.5),
            "ai_down_prob": float(sig.get("ai_down_prob") or 0.5),

            "rsi": float(sig.get("rsi") or 50),
            "vwap": float(sig.get("vwap") or price),

            "score": float(sig.get("score") or 0),

            "paper_trade": trade,
            "ai_trade_score": float(ai_score or 0.5),

            "bias": float(engine.bias or 0),

            "hot": ticker in hot_stocks
        }

    engine.update(market_prices)

    # 🧠 SAFETY FALLBACK
    if not results:
        return {
            "AAPL": {
                "ticker": "AAPL",
                "signal": "HOLD",
                "price": 0,
                "confidence": 0.5,
                "ai_up_prob": 0.5,
                "ai_down_prob": 0.5,
                "rsi": 50,
                "vwap": 0,
                "score": 0,
                "paper_trade": None,
                "ai_trade_score": 0.5,
                "bias": 0,
                "hot": False
            }
        }

    return results


# =========================================================
# HOT STOCKS
# =========================================================
@app.get("/hot")
def hot():
    return list(hot_stocks)


# =========================================================
# ON DEMAND STOCK
# =========================================================
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

        data = {ticker: {"df": df}}

        sig = alpha_engine(data, ai_predict)[ticker]

        stock_cache[ticker] = sig

        return sig

    except Exception as e:
        return {"error": str(e)}


# =========================================================
# STATS
# =========================================================
@app.get("/stats")
def stats():

    trades = engine.trades

    completed = [t for t in trades if t.get("result")]

    wins = len([t for t in completed if t["result"] == "WIN"])
    losses = len([t for t in completed if t["result"] == "LOSS"])

    total = len(completed)

    win_rate = wins / total if total else 0

    pnl = []

    for t in completed:
        if t.get("exit") is not None:
            if t["signal"] == "CALL":
                pnl.append(t["exit"] - t["entry"])
            else:
                pnl.append(t["entry"] - t["exit"])

    avg_pnl = sum(pnl) / len(pnl) if pnl else 0

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "bias": engine.bias,
        "avg_pnl": avg_pnl
    }