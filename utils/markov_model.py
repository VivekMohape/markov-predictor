import re
import random

# Define possible states
states = [
    "Ask for Technical Explanation",
    "Request Code Example",
    "Integrate/Optimize System",
    "Career or Relocation Question",
    "Learning/Certification Query",
    "Upload Another File",
    "Request Message/Email Draft",
    "Philosophical or Global Inquiry",
    "End/Acknowledge Conversation",
]

# Transition probabilities for each state (length matches 'states')
transition_matrix = {
    "Ask for Technical Explanation":          [0.35, 0.30, 0.15, 0.08, 0.04, 0.03, 0.03, 0.02, 0.00],
    "Request Code Example":                    [0.25, 0.10, 0.45, 0.05, 0.05, 0.05, 0.03, 0.02, 0.00],
    "Integrate/Optimize System":              [0.30, 0.20, 0.30, 0.10, 0.05, 0.03, 0.02, 0.00, 0.00],
    "Career or Relocation Question":          [0.15, 0.10, 0.10, 0.40, 0.15, 0.05, 0.03, 0.02, 0.00],
    "Learning/Certification Query":           [0.20, 0.10, 0.10, 0.15, 0.35, 0.05, 0.03, 0.02, 0.00],
    "Upload Another File":                     [0.40, 0.25, 0.10, 0.05, 0.05, 0.10, 0.03, 0.02, 0.00],
    "Request Message/Email Draft":            [0.30, 0.10, 0.10, 0.10, 0.05, 0.05, 0.25, 0.05, 0.00],
    "Philosophical or Global Inquiry":        [0.10, 0.10, 0.15, 0.10, 0.05, 0.05, 0.10, 0.35, 0.00],
    "End/Acknowledge Conversation":           [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.0],
}

def extract_keywords(text):
    """Detect important keywords in text."""
    text = text.lower()
    return {
        "ai": bool(re.search(r"\b(ai|artificial intelligence|machine learning)\b", text)),
        "data": bool(re.search(r"\b(data|analytics|science|big data)\b", text)),
        "job": bool(re.search(r"\b(job|career|work|employ|position)\b", text)),
        "research": bool(re.search(r"\b(research|paper|publication|study)\b", text)),
        "developer": bool(re.search(r"\b(developer|engineer|programmer|software)\b", text)),
        "student": bool(re.search(r"\b(student|university|college|bachelor|master)\b", text)),
    }

def predict_next_action(keywords):
    """Markov-based next-action prediction with safe probability handling."""
    
    # Determine current state based on detected keywords
    if any([keywords.get(k) for k in ("ai", "data", "developer")]):
        current_state = "Ask for Technical Explanation"
    elif keywords.get("research"):
        current_state = "Philosophical or Global Inquiry"
    elif keywords.get("student"):
        current_state = "Learning/Certification Query"
    elif keywords.get("job"):
        current_state = "Career or Relocation Question"
    else:
        current_state = random.choice(states)

    # Fetch probabilities safely
    probs = transition_matrix.get(current_state)
    if not probs:
        probs = [1.0 / len(states)] * len(states)
    elif len(probs) < len(states):
        probs += [0.0] * (len(states) - len(probs))
    
    # Predict next state
    next_state = random.choices(states, weights=probs, k=1)[0]
    
    return current_state, next_state, probs, states


# Example usage
if __name__ == "__main__":
    sample_text = "I am a student learning AI and data science."
    keywords = extract_keywords(sample_text)
    current, next_action, probs, all_states = predict_next_action(keywords)
    print("Current State:", current)
    print("Predicted Next State:", next_action)
    print("Probabilities:", probs)
