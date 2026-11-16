import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.model_selection import GroupKFold

from core import AnswerGrader, build_bank, clean_text, group_similar_mistakes, load_data, summarize

questions, responses, rubrics = load_data("data")
bank = build_bank(questions, rubrics)
grader = AnswerGrader("tfidf").fit(responses, bank)


def test_clean_text():
    assert clean_text("  a   b\n c ") == "a b c"
    assert clean_text(None) == ""


def test_blank_answer_is_not_graded():
    result = grader.predict_one("Q1", "   ")
    assert result["label"] == "blank"


def test_good_answer_beats_unrelated_answer():
    good = grader.predict_one("Q2", "DNS turns domain names into IP addresses.")
    bad = grader.predict_one("Q2", "It stores files on a hard disk.")
    assert good["features"]["reference_similarity"] > bad["features"]["reference_similarity"]


def test_unknown_question_raises():
    try:
        grader.predict_one("Q99", "anything")
        assert False
    except ValueError:
        pass


def test_variant_groups_never_split_across_folds():
    data = responses[responses["label"] != "blank"]
    for train_index, test_index in GroupKFold(5).split(data, groups=data["variant_group"]):
        train_groups = set(data.iloc[train_index]["variant_group"])
        test_groups = set(data.iloc[test_index]["variant_group"])
        assert not train_groups & test_groups


def test_summary_counts_blank_separately():
    table = pd.DataFrame({
        "question_id": ["Q1"] * 4,
        "predicted_label": ["correct", "partial", "incorrect", "blank"],
    })
    result = summarize(table, bank).iloc[0]
    assert result["graded"] == 3
    assert result["not_fully_correct_pct"] == 66.7


def test_mistake_groups_stay_within_one_question():
    table = pd.DataFrame({
        "question_id": ["Q2", "Q2", "Q3", "Q3"],
        "student_answer": ["DNS maps names to MAC addresses", "DNS maps names to MAC addresses",
                           "TCP never resends lost data", "TCP never resends lost data"],
        "predicted_label": ["incorrect"] * 4,
    })
    groups = group_similar_mistakes(table, grader.similarity)
    assert len(groups) == 2
    assert all(g["count"] == 2 for g in groups)


def test_api_predict_and_analyze():
    from main import app
    client = TestClient(app)
    assert len(client.get("/api/questions").json()) == len(bank)

    reply = client.post("/api/predict", json={"question_id": "Q1", "answer": "It separates network and host bits."})
    assert reply.status_code == 200
    assert reply.json()["label"] in ["correct", "partial", "incorrect"]

    csv_text = "response_id,question_id,student_answer\nA1,Q1,It marks the network part\nA2,Q2,\n"
    reply = client.post("/api/analyze", json={"csv_text": csv_text})
    assert reply.status_code == 200
    assert len(reply.json()["responses"]) == 2
    assert "mistake_groups" in reply.json()

    bad = client.post("/api/analyze", json={"csv_text": "id,text\n1,hi\n"})
    assert bad.status_code == 400
