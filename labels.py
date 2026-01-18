# labels.py

# =============================
# Guardrail taxonomy
# =============================
LABELS = [
    "SAFE",
    "PROMPT_INJECTION",
    # Future:
    # "PII",
    # "UNSAFE_INSTRUCTION",
]

# =============================
# Policy configuration
# =============================
POLICY_VERSION = "v1.0"

# Labels that should BLOCK when confident
BLOCKING_LABELS = {
    "PROMPT_INJECTION",
    # "PII",
    # "UNSAFE_INSTRUCTION",
}

# Severity per label
SEVERITY = {
    "SAFE": "none",
    "PROMPT_INJECTION": "critical",
    # "PII": "high",
    # "UNSAFE_INSTRUCTION": "critical",
}

# Confidence thresholds
LOW_CONF_THRESHOLD = 0.75       # Send to human review
BLOCK_THRESHOLD = 0.80          # Block request
