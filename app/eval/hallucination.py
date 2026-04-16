def hallucination_score(answer: str, context: str) -> float:
    """
    Simple heuristic:
    Measures how many answer words are unsupported by retrieved context.
    Lower is better.
    """
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    if not answer_words:
        return 0.0

    unsupported = answer_words - context_words
    score = len(unsupported) / max(len(answer_words), 1)
    return round(score, 3)


def hallucination_label(score: float) -> str:
    if score < 0.20:
        return "Low"
    elif score < 0.50:
        return "Medium"
    return "High"