from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd

from alpha_engine import alpha_engine
from ai_engine import ai_predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CACHE SYSTEM
# =========================
stock_cache = {}
hot_stocks = set()

TICKERS = [
    "AAPL","TSLA","NVDA","SPY","MSFT","AMZN","META","GOOGL",
    "AMD","NFLX","PLTR","INTC","COIN","NIO","RIVN","QQQ","RBLX"
]

# =========================
# MARKET DATA
# =========================
def get_market_data():

    data = {}

    for t in TICKERS:
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
# SCAN
# =========================
@app.get("/scan")
def scan():

    market = get_market_data()
    signals = alpha_engine(market, ai_predict)

    results = {}

    for ticker, sig in signals.items():

        confidence = sig.get("confidence", 0.5)

        if confidence > 0.75:
            hot_stocks.add(ticker)
            stock_cache[ticker] = sig

        results[ticker] = sig

    return results

# =========================
# HOT STOCKS
# =========================
@app.get("/hot")
def hot():
    return list(hot_stocks)

# =========================
# STOCK ON DEMAND
# =========================
@app.get("/stock/{ticker}")
def stock(ticker: str):

    ticker = ticker.upper()

    if ticker in stock_cache:
        return stock_cache[ticker]

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