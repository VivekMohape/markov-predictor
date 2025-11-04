import re
import random

def build_uniform_matrix(states):
    """Creates a uniform probability transition matrix."""
    n = len(states)
    if n == 0:
        return []
    matrix = []
    for i in range(n):
        probs = [1.0 / n] * n
        matrix.append(probs)
    return matrix


def _detect_intent(text):
    """Detect user intent based on text content."""
    t = text.lower()
    if re.search(r"\b(will|plan|next|future|forecast|goal|target|expect)\b", t):
        return "future"
    elif re.search(r"\b(since|last|previous|past|before)\b", t):
        return "past"
    elif re.search(r"\b(today|currently|now|present)\b", t):
        return "present"
    return "neutral"


def predict_next_state(states, matrix, current_state, text=None, deterministic=True):
    """Markov prediction aware of user intent and prevents self-loops."""
    result = {}
    try:
        if not states or current_state not in states:
            raise ValueError("Invalid or missing states.")

        idx = states.index(current_state)
        probs = matrix[idx][:]

        # Adjust based on intent
        if text:
            intent = _detect_intent(text)
            if intent == "future" and idx < len(states) - 1:
                idx += 1
                current_state = states[idx]
            elif intent == "past" and idx > 0:
                idx -= 1
                current_state = states[idx]

        # Remove self-loop unless final state
        if idx < len(states) - 1:
            probs[idx] = 0.0
        total = sum(probs)
        probs = [p / total for p in probs] if total > 0 else [1.0 / len(states)] * len(states)

        next_state = states[probs.index(max(probs))] if deterministic else random.choices(states, weights=probs, k=1)[0]

        result = {
            "current_state": current_state,
            "next_state": next_state,
            "probs": probs,
            "states": states,
        }

    except Exception as e:
        result["error"] = f"Markov prediction error: {e}"

    return result
