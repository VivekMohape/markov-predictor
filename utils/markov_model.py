import re
import random

# ------------------------------------------------------------
# 🧩 Utility Functions
# ------------------------------------------------------------
def build_uniform_matrix(states):
    """Create a uniform probability transition matrix for the given states."""
    n = len(states)
    if n == 0:
        return []
    matrix = []
    for i in range(n):
        # Initially equal probability to move to any state
        probs = [1.0 / n] * n
        matrix.append(probs)
    return matrix


def _detect_intent(text):
    """
    Detects user intent (past / present / future orientation)
    based on keywords in input text.
    """
    t = text.lower()
    if re.search(r"\b(will|plan|next|future|forecast|goal|target|expect)\b", t):
        return "future"
    elif re.search(r"\b(since|last|previous|past|before)\b", t):
        return "past"
    elif re.search(r"\b(today|currently|now|present)\b", t):
        return "present"
    return "neutral"


# ------------------------------------------------------------
# 🧮 Core Markov Predictor
# ------------------------------------------------------------
def predict_next_state(states, matrix, current_state, text=None, deterministic=True):
    """
    Predicts the next likely state based on current_state.
    - Removes self-loops unless it's the final state.
    - Optionally shifts state based on user intent (future/past).
    """
    result = {}
    try:
        if not states or current_state not in states:
            raise ValueError("Invalid state or missing state list.")

        idx = states.index(current_state)
        probs = matrix[idx][:]

        # Handle user intent (future, past, etc.)
        if text:
            intent = _detect_intent(text)
            if intent == "future":
                # Nudge the model forward one step
                if idx < len(states) - 1:
                    idx += 1
                    current_state = states[idx]
            elif intent == "past":
                # Nudge backward one step (for retrospective phrasing)
                if idx > 0:
                    idx -= 1
                    current_state = states[idx]
            # "present" or "neutral" -> no change

        # Remove self-loop probability except for final state
        if idx < len(states) - 1:
            probs[idx] = 0.0

        # Normalize probabilities
        total = sum(probs)
        if total == 0:
            probs = [1.0 / len(states)] * len(states)
        else:
            probs = [p / total for p in probs]

        # Choose next state deterministically or randomly
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
