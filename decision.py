def decision_logic(confidence):
    if confidence >= 0.75:
        return "High Confidence ✅"
    elif confidence >= 0.50:
        return "Needs Review ❓"
    else:
        return "Uncertain ⚠️"