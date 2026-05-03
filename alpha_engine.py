import pandas as pd

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def vwap(df):
    return (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()


def alpha_engine(market_data):

    signals = {}

    for ticker, data in market_data.items():

        df = data["df"]

        df["RSI"] = rsi(df["Close"])
        df["VWAP"] = vwap(df)

        price = df["Close"].iloc[-1]

     rsi_series = rsi(df["Close"])
     vwap_series = vwap(df)

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
            "price": float(price),
            "rsi": float(rsi_val),
            "vwap": float(vwap_val)
        }

    return signals