import math

def options_engine(df, signal):

    price = df["Close"].iloc[-1]

    # --- STRIKE SELECTION (nearest standard strike)
    strike_step = 5
    strike = round(price / strike_step) * strike_step

    # --- EXPIRY LOGIC (simple volatility-based)
    volatility = df["Close"].pct_change().std()

    if volatility > 0.02:
        expiry = "0DTE-1D"   # fast scalp
        style = "scalp"
    elif volatility > 0.01:
        expiry = "3-5D"
        style = "swing"
    else:
        expiry = "7-14D"
        style = "swing"

    # --- SIGNAL MAPPING
    if signal == "CALL":
        direction = "CALL"
    elif signal == "PUT":
        direction = "PUT"
    else:
        return {
            "strategy": "NONE",
            "strike": None,
            "expiry": None,
            "style": None
        }

    return {
        "strategy": direction,
        "strike": float(strike),
        "expiry": expiry,
        "style": style
    }