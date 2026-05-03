import math
import numpy as np

def options_engine(df, signal):

    try:
        if df is None or df.empty or "Close" not in df:
            return {
                "strategy": "NONE",
                "strike": None,
                "expiry": None,
                "style": None
            }

        close = df["Close"].dropna()

        if len(close) < 10:
            return {
                "strategy": "NONE",
                "strike": None,
                "expiry": None,
                "style": None
            }

        price = float(close.iloc[-1])

        # -------------------------
        # STRIKE SELECTION
        # -------------------------
        strike_step = 5
        strike = round(price / strike_step) * strike_step

        # -------------------------
        # VOLATILITY (SAFE)
        # -------------------------
        returns = close.pct_change().dropna()

        volatility = float(returns.std()) if len(returns) > 2 else 0.01

        # -------------------------
        # EXPIRY LOGIC
        # -------------------------
        if volatility > 0.02:
            expiry = "0DTE-1D"
            style = "scalp"

        elif volatility > 0.01:
            expiry = "3-5D"
            style = "swing"

        else:
            expiry = "7-14D"
            style = "swing"

        # -------------------------
        # SIGNAL MAPPING
        # -------------------------
        if signal not in ["CALL", "PUT"]:
            return {
                "strategy": "NONE",
                "strike": None,
                "expiry": None,
                "style": None
            }

        return {
            "strategy": signal,
            "strike": float(strike),
            "expiry": expiry,
            "style": style
        }

    except Exception:
        return {
            "strategy": "NONE",
            "strike": None,
            "expiry": None,
            "style": None
        }