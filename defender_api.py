from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

MODEL_PATH = "./guardrail_model"
CONFIDENCE_THRESHOLD = 0.8

app = FastAPI(title="Guardrail Defender")

classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)

class GuardrailRequest(BaseModel):
    text: str

class GuardrailResponse(BaseModel):
    label: str
    confidence: float
    decision: str

@app.post("/guardrail", response_model=GuardrailResponse)
def guardrail_check(req: GuardrailRequest):
    result = classifier(req.text)[0]

    label = result["label"]
    confidence = round(result["score"], 3)

    decision = (
        "BLOCK"
        if label == "PROMPT_INJECTION" and confidence >= CONFIDENCE_THRESHOLD
        else "ALLOW"
    )

    return {
        "label": label,
        "confidence": confidence,
        "decision": decision
    }
