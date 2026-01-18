from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sqlite3
from datetime import datetime
import pandas as pd
from transformers import pipeline

MODEL_PATH = "./guardrail_model"
DB_NAME = "chat_memory.db"

app = FastAPI()

# Serve frontend files from /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html on root
@app.get("/")
def home():
    return FileResponse("static/index.html")


# ---------------- DB SETUP ----------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            reviewer_label TEXT DEFAULT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            role TEXT,
            content TEXT,
            model_label TEXT,
            model_score REAL,
            timestamp TEXT,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
    """)

    conn.commit()
    conn.close()

init_db()


# ---------------- MODEL LOAD ONCE ----------------
clf = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)


# ---------------- REQUEST SCHEMAS ----------------
class StartResponse(BaseModel):
    conversation_id: int

class MessageRequest(BaseModel):
    conversation_id: int
    user_prompt: str

class ReviewRequest(BaseModel):
    reviewer_label: str  # "SAFE" or "PROMPT_INJECTION"


# ---------------- HELPERS ----------------
def insert_message(conversation_id, role, content, model_label=None, model_score=None):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO messages (conversation_id, role, content, model_label, model_score, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        conversation_id,
        role,
        content,
        model_label,
        model_score,
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()


# ---------------- API ----------------
@app.post("/api/start", response_model=StartResponse)
def start_conversation():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("INSERT INTO conversations (timestamp) VALUES (?)", (datetime.utcnow().isoformat(),))
    convo_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"conversation_id": convo_id}


@app.get("/api/conversations")
def list_conversations():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, reviewer_label FROM conversations ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()

    return {
        "conversations": [
            {"conversation_id": r[0], "timestamp": r[1], "reviewer_label": r[2]}
            for r in rows
        ]
    }


@app.post("/api/message")
def send_message(req: MessageRequest):
    # Store user prompt
    insert_message(req.conversation_id, "user", req.user_prompt)

    # Model prediction
    result = clf(req.user_prompt)[0]
    label = result["label"]
    score = float(result["score"])

    # Store assistant response
    assistant_reply = f"🛡️ Prediction: {label} (score={score:.4f})"
    insert_message(req.conversation_id, "assistant", assistant_reply, model_label=label, model_score=score)

    return {
        "model_label": label,
        "model_score": score,
        "assistant_reply": assistant_reply
    }


@app.get("/api/conversation/{conversation_id}")
def get_conversation(conversation_id: int):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        SELECT role, content, model_label, model_score, timestamp
        FROM messages
        WHERE conversation_id=?
        ORDER BY id ASC
    """, (conversation_id,))
    rows = cur.fetchall()
    conn.close()

    return {
        "conversation_id": conversation_id,
        "messages": [
            {
                "role": r[0],
                "content": r[1],
                "model_label": r[2],
                "model_score": r[3],
                "timestamp": r[4]
            }
            for r in rows
        ]
    }


@app.post("/api/review/{conversation_id}")
def review_conversation(conversation_id: int, req: ReviewRequest):
    if req.reviewer_label not in ["SAFE", "PROMPT_INJECTION"]:
        return {"error": "reviewer_label must be SAFE or PROMPT_INJECTION"}

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        UPDATE conversations
        SET reviewer_label=?
        WHERE id=?
    """, (req.reviewer_label, conversation_id))
    conn.commit()
    conn.close()

    return {"status": "updated", "conversation_id": conversation_id, "reviewer_label": req.reviewer_label}


@app.get("/api/export/excel")
def export_excel():
    conn = sqlite3.connect(DB_NAME)
    df_convos = pd.read_sql_query("SELECT * FROM conversations ORDER BY id DESC", conn)
    df_msgs = pd.read_sql_query("SELECT * FROM messages ORDER BY id DESC", conn)
    conn.close()

    filename = "guardrail_memory.xlsx"
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_convos.to_excel(writer, index=False, sheet_name="conversations")
        df_msgs.to_excel(writer, index=False, sheet_name="messages")

    return FileResponse(
        filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename
    )
