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
MODEL_PATH = BASE_DIR / "models" / "tfidf.joblib"

app = FastAPI(title="Answer Sheet Analyzer")


def load_grader():
    """Load the saved model. If it is missing, train one from the sample data."""
    if MODEL_PATH.exists():
        return AnswerGrader.load(MODEL_PATH)
    questions, responses, rubrics = load_data(str(BASE_DIR / "data"))
    bank = build_bank(questions, rubrics)
    grader = AnswerGrader("tfidf").fit(responses, bank)
    grader.save(MODEL_PATH)
    return grader


grader = load_grader()


def to_records(table):
    """DataFrame -> list of dicts, with missing numbers as None (valid JSON)."""
    table = table.astype(object).where(table.notna(), None)
    return table.to_dict("records")


class AnswerIn(BaseModel):
    question_id: str
    answer: str


class CsvIn(BaseModel):
    csv_text: str


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/questions")
def list_questions():
    return [
        {
            "question_id": qid,
            "question": info["question"],
            "topic": info["topic"],
            "model_answer": info["model_answer"],
        }
        for qid, info in grader.bank.items()
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
        return grader.predict_one(item.question_id, item.answer)
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

    try:
        predictions = grader.predict_table(data)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    rows = predictions[["response_id", "question_id", "student_answer",
                        "predicted_label", "reference_similarity"]].round(3)
    return {
        "responses": to_records(rows),
        "by_question": to_records(summarize(predictions, grader.bank, "question_id")),
        "by_topic": to_records(summarize(predictions, grader.bank, "topic")),
        "mistake_groups": group_similar_mistakes(predictions, grader.similarity),
    }
