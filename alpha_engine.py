import pandas as pd

import numpy as np

def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def vwap(df):
    return (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()


def alpha_engine(market_data):

    signals = {}

    for ticker, data in market_data.items():

        df = data["df"]

        if df is None or df.empty:
            signals[ticker] = {"signal": "NO_DATA"}
            continue

        df = df.dropna()

        rsi_series = rsi(df["Close"])
        vwap_series = vwap(df)

        price = float(df["Close"].iloc[-1])
        rsi_val = float(rsi_series.iloc[-1])
        vwap_val = float(vwap_series.iloc[-1])

        momentum = float(df["Close"].iloc[-1] - df["Close"].iloc[-5])

        score = 0

        if price > vwap_val:
            score += 1
        else:
            score -= 1

        if rsi_val < 35:
            score += 2
        elif rsi_val > 70:
            score -= 2

        if momentum > 0:
            score += 1
        else:
            score -= 1

        if score >= 2:
            signal = "CALL"
        elif score <= -2:
            signal = "PUT"
        else:
            signal = "HOLD"

        signals[ticker] = {
            "signal": signal,
            "score": score,
            "price": price,
            "rsi": rsi_val,
            "vwap": vwap_val
        }

    return signals
    def last(x):
return float(x.dropna().iloc[-1])

price = last(df["Close"])
rsi_val = last(rsi_series)
vwap_val = last(vwap_series)       