"""FastAPI backend.

    python train.py
    python -m uvicorn main:app --reload
    open http://127.0.0.1:8000
"""
import io
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core import AnswerGrader, build_bank, group_similar_mistakes, load_data, summarize

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(title="Answer Sheet Analyzer")

graders: dict[str, AnswerGrader] = {}


def get_grader(backend: str = "tfidf") -> AnswerGrader:
    """Load the model for the requested backend (tfidf or sbert), caching it in memory."""
    if backend not in ("tfidf", "sbert"):
        backend = "tfidf"
    if backend not in graders:
        model_path = MODELS_DIR / f"{backend}.joblib"
        if model_path.exists():
            graders[backend] = AnswerGrader.load(model_path)
        else:
            questions, responses, rubrics = load_data(str(BASE_DIR / "data"))
            bank = build_bank(questions, rubrics)
            g = AnswerGrader(backend).fit(responses, bank)
            g.save(model_path)
            graders[backend] = g
    return graders[backend]


# Preload the default TF-IDF grader
grader = get_grader("tfidf")


def to_records(table):
    """DataFrame -> list of dicts, with missing numbers as None (valid JSON)."""
    table = table.astype(object).where(table.notna(), None)
    return table.to_dict("records")


class AnswerIn(BaseModel):
    question_id: str
    answer: str
    backend: str = "tfidf"


class CsvIn(BaseModel):
    csv_text: str
    backend: str = "tfidf"


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/questions")
def list_questions():
    bank = get_grader("tfidf").bank
    return [
        {
            "question_id": qid,
            "question": info["question"],
            "topic": info["topic"],
            "model_answer": info["model_answer"],
        }
        for qid, info in bank.items()
    ]


@app.get("/api/demo-csv")
def get_demo_csv():
    demo_file = BASE_DIR / "data" / "responses.csv"
    if not demo_file.exists():
        raise HTTPException(status_code=404, detail="Demo responses file not found.")
    return {"csv_text": demo_file.read_text(encoding="utf-8")}


@app.post("/api/predict")
def predict(item: AnswerIn):
    try:
        current_grader = get_grader(item.backend)
        res = current_grader.predict_one(item.question_id, item.answer)
        res["backend"] = item.backend
        return res
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/api/analyze")
def analyze(item: CsvIn):
    """Grade a whole CSV (response_id, question_id, student_answer)."""
    try:
        data = pd.read_csv(io.StringIO(item.csv_text), dtype=str, keep_default_na=False)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the CSV text.")

    missing = {"response_id", "question_id", "student_answer"} - set(data.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {sorted(missing)}")
    if data.empty:
        raise HTTPException(status_code=400, detail="The CSV has no rows.")

    current_grader = get_grader(item.backend)
    try:
        predictions = current_grader.predict_table(data)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    rows = predictions[["response_id", "question_id", "student_answer",
                        "predicted_label", "reference_similarity"]].round(3)
    return {
        "backend": item.backend,
        "responses": to_records(rows),
        "by_question": to_records(summarize(predictions, current_grader.bank, "question_id")),
        "by_topic": to_records(summarize(predictions, current_grader.bank, "topic")),
        "mistake_groups": group_similar_mistakes(predictions, current_grader.similarity),
    }
