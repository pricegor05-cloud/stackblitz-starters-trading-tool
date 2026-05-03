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

    return signals