def safe_ai(df):
    try:
        ai = ai_predict(df)

        if not isinstance(ai, dict):
            ai = {}

        up = float(ai.get("ai_up_prob", 0.5))
        down = float(ai.get("ai_down_prob", 0.5))
        conf = float(ai.get("confidence", 0.5))

        # 🧠 normalize (true probability space)
        total = up + down
        if total > 0:
            up /= total
            down /= total

        return {
            "ai_up_prob": up,
            "ai_down_prob": down,
            "confidence": conf
        }

    except Exception:
        return {
            "ai_up_prob": 0.5,
            "ai_down_prob": 0.5,
            "confidence": 0.5
            
            # 🧠 STEP 4: AI CONFIDENCE CALIBRATION

ai_accuracy = ai_brain.accuracy()

# smooth scaling (prevents extreme swings)
confidence = float(
    min(
        1.0,
        confidence * (0.7 + ai_accuracy)
    )
)
        }
        class AIBrain:
    def __init__(self):
        self.memory = []  # (prediction, result)

    def log(self, pred, result):
        self.memory.append((pred, result))

    def accuracy(self):
        if len(self.memory) < 10:
            return 0.5

        correct = 0
        total = 0

        for p, r in self.memory:
            if r is None:
                continue

            if (p > 0.5 and r == "WIN") or (p < 0.5 and r == "LOSS"):
                correct += 1

            total += 1

        return correct / total if total else 0.5