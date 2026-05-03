import numpy as np

def detect_market_regime(df):

    try:
        if df is None or df.empty or "Close" not in df:
            return "RANGING"

        close = df["Close"].dropna()

        if len(close) < 20:
            return "RANGING"

        returns = close.pct_change().dropna()

        volatility = returns.std()

        momentum = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]

        if volatility > 0.025:
            return "HIGH_VOLATILITY"

        if abs(momentum) > 0.03:
            return "TRENDING"

        return "RANGING"

    except Exception:
        return "RANGING"