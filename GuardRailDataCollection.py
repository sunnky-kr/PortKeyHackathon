import uuid
import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from labels import (
    LABELS,
    BLOCKING_LABELS,
    SEVERITY,
    LOW_CONF_THRESHOLD,
    BLOCK_THRESHOLD,
    POLICY_VERSION
)

REVIEW_FILE = "review_queue.jsonl"

app = FastAPI()

clf = pipeline(
    "text-classification",
    model="./guardrail_model",
    tokenizer="./guardrail_model"
)

class GuardrailRequest(BaseModel):
    text: str


class GuardrailResponse(BaseModel):
    label: str
    confidence: float
    severity: str
    action: str
    policy_version: str


def log_for_review(text, label, confidence):
    record = {
        "id": str(uuid.uuid4()),
        "text": text,
        "model_label": label,
        "confidence": confidence,
        "severity": SEVERITY.get(label, "unknown"),
        "policy_version": POLICY_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "human_label": None,
        "available_labels": LABELS
    }
    with open(REVIEW_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def decide_action(label, confidence):
    if label in BLOCKING_LABELS and confidence >= BLOCK_THRESHOLD:
        return "BLOCK"
    return "ALLOW"


@app.post("/guardrail", response_model=GuardrailResponse)
def guardrail(req: GuardrailRequest):
    result = clf(req.text)[0]
    label = result["label"]
    confidence = float(result["score"])

    # Low-confidence → human review
    if confidence < LOW_CONF_THRESHOLD:
        log_for_review(req.text, label, confidence)

    action = decide_action(label, confidence)

    return {
        "label": label,
        "confidence": confidence,
        "severity": SEVERITY.get(label, "unknown"),
        "action": action,
        "policy_version": POLICY_VERSION
    }
