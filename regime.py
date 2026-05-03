import numpy as np

def detect_market_regime(df):

    close = df["Close"]

    returns = close.pct_change().dropna()

    volatility = returns.std()

    momentum = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]

    rsi = 50  # fallback if not passed in (you can improve later)

    # -------------------------
    # REGIME RULES
    # -------------------------

    if volatility > 0.02:
        return "HIGH_VOLATILITY"

    if abs(momentum) > 0.03:
        return "TRENDING"

    return "RANGING"