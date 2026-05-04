from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd

from alpha_engine import alpha_engine
from ai_engine import ai_predict
from paper_engine import PaperEngine

app = FastAPI()
engine = PaperEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# V10 GLOBAL STATE
# =========================
CACHE = {}
HOT = set()

TICKERS = [
    "AAPL","TSLA","NVDA","SPY","MSFT","AMZN","META","GOOGL",
    "AMD","NFLX","PLTR","INTC","COIN","NIO","RIVN","QQQ","AVGO","RBLX"
]

# =========================
# SAFE DATA FETCH
# =========================
def fetch_data(ticker):

    df = yf.download(ticker, period="5d", interval="5m", progress=False)

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()

    if len(df) < 20:
        return None

    return df

# =========================
# SNAPSHOT ENGINE (CORE V10)
# =========================
def build_snapshot():

    market = {}
    signals = alpha_engine(
        {t: {"df": fetch_data(t)} for t in TICKERS if fetch_data(t) is not None},
        ai_predict
    )

    for t, sig in signals.items():

        if not isinstance(sig, dict):
            continue

        price = float(sig.get("price", 0))
        confidence = float(sig.get("confidence", 0.5))

        # 🧠 hot detection
        if confidence > 0.80:
            HOT.add(t)

        trade = engine.execute(
            t,
            sig.get("signal", "HOLD"),
            price,
            sig.get("options", {})
        )

        market[t] = {
            "ticker": t,
            "signal": sig.get("signal", "HOLD"),
            "price": price,
            "confidence": confidence,
            "ai_up_prob": sig.get("ai_up_prob", 0.5),
            "ai_down_prob": sig.get("ai_down_prob", 0.5),
            "rsi": sig.get("rsi", 50),
            "vwap": sig.get("vwap", price),
            "trade": trade,
            "hot": t in HOT
        }

    return market


# =========================
# GLOBAL SNAPSHOT (FAST)
# =========================
SNAPSHOT = build_snapshot()


@app.get("/scan")
def scan():
    global SNAPSHOT
    SNAPSHOT = build_snapshot()
    return SNAPSHOT


@app.get("/hot")
def hot():
    return list(HOT)


@app.get("/stock/{ticker}")
def stock(ticker: str):

    ticker = ticker.upper()

    if ticker in CACHE:
        return CACHE[ticker]

    df = fetch_data(ticker)

    if df is None:
        return {"error": "no data"}

    sig = alpha_engine({ticker: {"df": df}}, ai_predict)[ticker]

    CACHE[ticker] = sig
    return sig