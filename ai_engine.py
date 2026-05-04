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

        returns = np.diff(close)

        momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        volatility = np.std(returns[-10:]) if len(returns) >= 10 else 0

        trend = close[-1] - close[-10] if len(close) >= 10 else 0

        score = 0
        score += 1 if trend > 0 else -1
        score += 1 if momentum > 0 else -1
        score += 1 if volatility < np.std(close) * 0.5 else -1

        ai_up = 1 / (1 + np.exp(-score))
        ai_down = 1 - ai_up

        confidence = abs(ai_up - 0.5) * 2

        return {
            "ai_up_prob": float(ai_up),
            "ai_down_prob": float(ai_down),
            "confidence": float(confidence)
        }

    except:
        return {
            "ai_up_prob": 0.5,
            "ai_down_prob": 0.5,
            "confidence": 0.0
        }