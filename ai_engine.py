import numpy as np

def ai_predict(df):

    try:
        close = df["Close"].dropna().values

        if len(close) < 20:
            return {
                "ai_up_prob": 0.5,
                "ai_down_prob": 0.5,
                "confidence": 0.0
            }

        # -------------------------
        # SAFE RETURNS
        # -------------------------
        returns = np.diff(close)

        # safe slicing
        last_5 = returns[-5:] if len(returns) >= 5 else returns
        last_10 = returns[-10:] if len(returns) >= 10 else returns

        momentum = np.mean(last_5)
        volatility = np.std(last_10)

        # trend (safe index)
        trend = close[-1] - close[-10]

        score = 0

        # -------------------------
        # SIGNAL ENGINE
        # -------------------------
        score += 1 if trend > 0 else -1
        score += 1 if momentum > 0 else -1

        if volatility < np.std(close) * 0.5:
            score += 1
        else:
            score -= 0.5

        # -------------------------
        # PROBABILITY MAPPING
        # -------------------------
        ai_up = 1 / (1 + np.exp(-score))
        ai_down = 1 - ai_up

        confidence = abs(ai_up - 0.5) * 2

        return {
            "ai_up_prob": float(ai_up),
            "ai_down_prob": float(ai_down),
            "confidence": float(confidence)
        }

    except Exception:
        return {
            "ai_up_prob": 0.5,
            "ai_down_prob": 0.5,
            "confidence": 0.0
        }