def safe_ai(df):
    try:
        ai = ai_predict(df)

        # 🧠 enforce clean numeric outputs
        return {
            "ai_up_prob": float(ai.get("ai_up_prob", 0.5)),
            "ai_down_prob": float(ai.get("ai_down_prob", 0.5)),
            "confidence": float(ai.get("confidence", 0.5))
        }

    except Exception:
        return {
            "ai_up_prob": 0.5,
            "ai_down_prob": 0.5,
            "confidence": 0.5
        }