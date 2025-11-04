# utils/markov_model.py
import random
import logging
logger = logging.getLogger(__name__)

def build_uniform_matrix(states):
    """Create a simple uniform transition matrix for unseen domains."""
    n = len(states)
    if n == 0:
        raise ValueError("No states supplied.")
    p = 1.0 / n
    base = [p] * n
    return {s: base[:] for s in states}

def predict_next_state(states, matrix, current_state, deterministic=True):
    """
    Pure Markov predictor.
    - states: list of state names
    - matrix: dict mapping state -> probability list
    - current_state: current state's name
    """
    try:
        probs = matrix.get(current_state)
        if not probs or len(probs) != len(states):
            probs = [1.0 / len(states)] * len(states)
        if deterministic:
            next_state = states[probs.index(max(probs))]
        else:
            next_state = random.choices(states, weights=probs, k=1)[0]
        return {"current_state": current_state, "next_state": next_state,
                "probs": probs, "states": states}
    except Exception as e:
        logger.exception("Markov prediction error")
        return {"error": str(e)}
