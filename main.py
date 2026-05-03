def get_market_data():
    tickers = ["AAPL", "TSLA", "NVDA", "SPY"]

    data = {}

    for t in tickers:
        df = yf.download(t, period="5d", interval="5m")
df = df.copy()

# flatten possible MultiIndex from yfinance
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
        if df is None or df.empty:
            continue

        data[t] = {
            "df": df
        }

    return data
from alpha_engine import alpha_engine
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()

# allow StackBlitz
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/scan")
def scan():

    market = get_market_data()

    signals = alpha_engine(market)

    return signals