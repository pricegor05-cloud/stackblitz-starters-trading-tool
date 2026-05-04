from paper_engine import PaperEngine
import yfinance as yf
import pandas as pd
from alpha_engine import alpha_engine
from ai_engine import ai_predict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from paper_engine import PaperEngine, ExecutionLayer

engine = PaperEngine()

exec_layer = ExecutionLayer()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GLOBAL SYSTEM MEMORY
# =========================
stock_cache = {}
hot_stocks = set()

WATCHLIST = [
    "AAPL","TSLA","NVDA","SPY","MSFT","AMZN","META","GOOGL",
    "AMD","NFLX","PLTR","INTC","COIN","NIO","RIVN","QQQ","AVGO"
]

# =========================
# MARKET DATA (SAFE)
# =========================
def get_market_data():
    data = {}

    for t in WATCHLIST:
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

        except:
            continue

    return data


# =========================
# SCAN ENGINE
# =========================
@app.get("/scan")
def scan():

    market = get_market_data()
    signals = alpha_engine(market, ai_predict)

    results = {}
    market_prices = {}

    for ticker, sig in signals.items():

        if not isinstance(sig, dict):
            continue

        price = float(sig.get("price", 0))
        confidence = float(sig.get("confidence", 0.5))
        ai_up = float(sig.get("ai_up_prob", 0.5))
        ai_down = float(sig.get("ai_down_prob", 0.5))

        market_prices[ticker] = price

        # 🔥 HOT STOCK LOGIC
        if confidence > 0.78:
            hot_stocks.add(ticker)
            stock_cache[ticker] = sig

        # =========================
        # V12 EXECUTION LAYER (FIXED)
        # =========================
        trade = None

        if (
            confidence > 0.72
            and sig.get("signal") in ["CALL", "PUT"]
        ):

            if (sig["signal"] == "CALL" and ai_up > 0.55) or \
               (sig["signal"] == "PUT" and ai_down > 0.55):

                trade = exec_layer.enter(
                    ticker,
                    sig["signal"],
                    price,
                    confidence
                )

        # 🧠 AI SCORE
        ai_score = engine.predict_success(
            sig.get("signal", "HOLD"),
            price,
            sig.get("rsi", 50),
            sig.get("vwap", price)
        )

        if trade:
            trade["ai_score"] = ai_score

        # =========================
        # CLEAN OUTPUT (UI SAFE)
        # =========================
        results[ticker] = {
            "ticker": ticker,
            "signal": sig.get("signal", "HOLD"),
            "price": price,

            "confidence": confidence,
            "ai_up_prob": ai_up,
            "ai_down_prob": ai_down,

            "rsi": float(sig.get("rsi", 50)),
            "vwap": float(sig.get("vwap", price)),

            "score": float(sig.get("score", 0)),

            "paper_trade": trade,
            "ai_trade_score": float(ai_score),

            "hot": ticker in hot_stocks
        }

    engine.update(market_prices)
    exec_layer.update(market_prices)

    return results


# =========================
# HOT STOCKS
# =========================
@app.get("/hot")
def hot():
    return list(hot_stocks)


# =========================
# ON DEMAND STOCK
# =========================
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

        sig = alpha_engine({ticker: {"df": df}}, ai_predict)[ticker]

        stock_cache[ticker] = sig

        return sig

    except:
        return {"error": "failed"}