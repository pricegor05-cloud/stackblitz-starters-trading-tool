from paper_engine import PaperEngine, ExecutionLayer
import yfinance as yf
import pandas as pd
from alpha_engine import alpha_engine
from ai_engine import ai_predict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

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
# GLOBAL STATE (V13 CORE)
# =========================
stock_cache = {}
hot_stocks = set()

portfolio = {
    "balance": 10000,
    "equity_curve": [],
    "open_positions": {},
    "closed_trades": []
}

WATCHLIST = [
    "AAPL","TSLA","NVDA","SPY","MSFT","AMZN","META","GOOGL",
    "AMD","NFLX","PLTR","INTC","COIN","NIO","RIVN","QQQ","AVGO"
]

# =========================
# MARKET DATA
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
# POSITION TRACKING (V13)
# =========================
def update_portfolio(market_prices):

    for ticker, pos in list(portfolio["open_positions"].items()):

        price = market_prices.get(ticker)
        if price is None:
            continue

        entry = pos["entry"]

        # simple PnL logic
        if pos["signal"] == "CALL":
            pnl = (price - entry)
        else:
            pnl = (entry - price)

        pos["pnl"] = pnl

        # exit logic
        if pnl <= -1.0:
            pos["status"] = "LOSS"
            portfolio["closed_trades"].append(pos)
            del portfolio["open_positions"][ticker]

        elif pnl >= 2.0:
            pos["status"] = "WIN"
            portfolio["closed_trades"].append(pos)
            del portfolio["open_positions"][ticker]


# =========================
# SCAN ENGINE (V13 CORE)
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

        market_prices[ticker] = price

        # HOT STOCKS
        if confidence > 0.78:
            hot_stocks.add(ticker)
            stock_cache[ticker] = sig

        trade = None

        try:
            if confidence > 0.72 and sig.get("signal") in ["CALL", "PUT"]:

                if ticker not in portfolio["open_positions"]:

                    trade = exec_layer.enter(
                        ticker,
                        sig["signal"],
                        price,
                        confidence
                    )

                    if trade:
                        portfolio["open_positions"][ticker] = {
                            "ticker": ticker,
                            "signal": sig["signal"],
                            "entry": price,
                            "time": str(datetime.utcnow()),
                            "status": "OPEN",
                            "pnl": 0
                        }

        except:
            trade = None

        ai_score = engine.predict_success(
            sig.get("signal", "HOLD"),
            price,
            sig.get("rsi", 50),
            sig.get("vwap", price)
        )

        results[ticker] = {
            "ticker": ticker,
            "signal": sig.get("signal", "HOLD"),
            "price": price,
            "confidence": confidence,

            "ai_up_prob": sig.get("ai_up_prob", 0.5),
            "ai_down_prob": sig.get("ai_down_prob", 0.5),

            "rsi": sig.get("rsi", 50),
            "vwap": sig.get("vwap", price),
            "score": sig.get("score", 0),

            "paper_trade": trade,
            "ai_trade_score": float(ai_score),

            "hot": ticker in hot_stocks
        }

    engine.update(market_prices)
    exec_layer.update(market_prices)

    update_portfolio(market_prices)

    # NEVER EMPTY RESPONSE
    if not results:
        return {
            "AAPL": {
                "ticker": "AAPL",
                "signal": "HOLD",
                "price": 0,
                "confidence": 0.5,
                "ai_up_prob": 0.5,
                "ai_down_prob": 0.5,
                "paper_trade": None,
                "hot": False
            }
        }

    return results


# =========================
# HOT STOCKS
# =========================
@app.get("/hot")
def hot():
    return list(hot_stocks)


# =========================
# PORTFOLIO (NEW V13 FEATURE)
# =========================
@app.get("/portfolio")
def get_portfolio():
    return portfolio


# =========================
# EQUITY CURVE (NEW V13 FEATURE)
# =========================
@app.get("/equity")
def equity():

    total_pnl = sum([
        t.get("pnl", 0)
        for t in portfolio["closed_trades"]
    ])

    portfolio["equity_curve"].append({
        "time": str(datetime.utcnow()),
        "equity": 10000 + total_pnl
    })

    return portfolio["equity_curve"]


# =========================
# STOCK ENDPOINT
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