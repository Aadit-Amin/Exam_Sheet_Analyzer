"""Core logic for the Answer Sheet Analyzer.

Idea: compare a student's answer with the teacher's model answer and key
points, turn that into a few numbers (features), and let a small
classifier decide correct / partial / incorrect.
"""
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABELS = ["incorrect", "partial", "correct"]
FEATURE_NAMES = ["reference_similarity", "keypoint_mean", "keypoint_min", "length_ratio"]


# ---------------------------------------------------------------- data

def clean_text(text):
    """Remove extra spaces/newlines. Missing values become an empty string."""
    if text is None or pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def load_data(data_dir="data"):
    """Read the three data files and return (questions, responses, rubrics)."""
    questions = pd.read_csv(f"{data_dir}/questions.csv")
    responses = pd.read_csv(f"{data_dir}/responses.csv", keep_default_na=False)
    with open(f"{data_dir}/rubrics.json", encoding="utf-8") as f:
        rubrics = json.load(f)
    return questions, responses, rubrics


def build_bank(questions, rubrics):
    """Put everything about each question in one dictionary.

    bank["Q1"] = {"question": ..., "model_answer": ..., "topic": ..., "key_points": [...]}
    """
    bank = {}
    for _, row in questions.iterrows():
        qid = row["question_id"]
        bank[qid] = {
            "question": clean_text(row["question"]),
            "model_answer": clean_text(row["model_answer"]),
            "topic": clean_text(row["topic"]),
            "key_points": rubrics[qid]["key_points"],
        }
    return bank


# ---------------------------------------------------------------- similarity

class TextSimilarity:
    """Cosine similarity between two texts.

    backend="tfidf": word counts weighted by importance (needs no download).
    backend="sbert": sentence embeddings from all-MiniLM-L6-v2 (optional,
                     downloads the model the first time).
    """

    def __init__(self, backend="tfidf"):
        if backend not in ("tfidf", "sbert"):
            raise ValueError("backend must be 'tfidf' or 'sbert'")
        self.backend = backend
        self.vectorizer = None
        self.encoder = None
        self.cache = {}

    def fit(self, texts):
        """TF-IDF learns its vocabulary here. SBERT needs no fitting."""
        if self.backend == "tfidf":
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            self.vectorizer.fit(texts)
        self.cache = {}
        return self

    def embed(self, text):
        """Turn a text into a vector (remembered so we do not repeat work)."""
        if text in self.cache:
            return self.cache[text]
        if self.backend == "tfidf":
            vector = self.vectorizer.transform([text]).toarray()[0]
        else:
            if self.encoder is None:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            vector = self.encoder.encode([text], normalize_embeddings=True)[0]
        self.cache[text] = vector
        return vector

    def score(self, text_a, text_b):
        """Cosine similarity between 0 and 1 (0 if either text has no known words)."""
        a = self.embed(text_a)
        b = self.embed(text_b)
        size = np.linalg.norm(a) * np.linalg.norm(b)
        if size == 0:
            return 0.0
        return float(np.dot(a, b) / size)

    def __getstate__(self):
        # The SBERT model is big, so it is not saved with the grader.
        state = self.__dict__.copy()
        state["encoder"] = None
        state["cache"] = {}
        return state


# ---------------------------------------------------------------- features

def split_sentences(answer):
    parts = re.split(r"(?<=[.!?])\s+|\n+", answer)
    return [p.strip() for p in parts if p.strip()]


def get_features(answer, info, similarity):
    """Make the 4 numbers for one answer. Also return evidence for each key point.

    reference_similarity: similarity to the model answer
    keypoint_mean:        average of the best match found for each key point
    keypoint_min:         the weakest key point (missing a point lowers this)
    length_ratio:         answer length / model answer length, capped at 3
    """
    reference_similarity = similarity.score(answer, info["model_answer"])

    # A key point can match any single sentence or the whole answer.
    candidates = split_sentences(answer) + [answer]
    keypoint_scores = []
    evidence = []
    for key_point in info["key_points"]:
        best_score = 0.0
        best_sentence = ""
        for sentence in candidates:
            score = similarity.score(key_point, sentence)
            if score > best_score:
                best_score = score
                best_sentence = sentence
        keypoint_scores.append(best_score)
        evidence.append({
            "key_point": key_point,
            "similarity": round(best_score, 3),
            "matched_text": best_sentence,
        })

    answer_words = len(answer.split())
    reference_words = max(1, len(info["model_answer"].split()))
    length_ratio = min(answer_words / reference_words, 3.0)

    features = [reference_similarity, float(np.mean(keypoint_scores)),
                float(np.min(keypoint_scores)), length_ratio]
    return features, evidence


# ---------------------------------------------------------------- grader

class AnswerGrader:
    """Fit on labelled answers, then predict labels for new answers."""

    def __init__(self, backend="tfidf"):
        self.backend = backend
        self.similarity = TextSimilarity(backend)
        self.classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        )
        self.bank = None

    def fit(self, train, bank):
        """train: DataFrame with question_id, student_answer, label."""
        self.bank = bank
        train = train[train["label"] != "blank"]

        # TF-IDF vocabulary comes only from training answers and teacher text.
        corpus = [clean_text(a) for a in train["student_answer"]]
        for info in bank.values():
            corpus.append(info["model_answer"])
            corpus.extend(info["key_points"])
        self.similarity.fit(corpus)

        X, _ = self.feature_matrix(train)
        self.classifier.fit(X, train["label"])
        return self

    def feature_matrix(self, data):
        """Return (numbers for every row, evidence for every row)."""
        rows = []
        all_evidence = []
        for _, row in data.iterrows():
            qid = row["question_id"]
            if qid not in self.bank:
                raise ValueError(f"Unknown question_id: {qid}")
            answer = clean_text(row["student_answer"])
            features, evidence = get_features(answer, self.bank[qid], self.similarity)
            rows.append(features)
            all_evidence.append(evidence)
        return np.array(rows), all_evidence

    def predict_table(self, data):
        """Add predicted_label and the feature columns to a copy of data."""
        result = data.copy().reset_index(drop=True)
        result["predicted_label"] = "blank"
        for name in FEATURE_NAMES:
            result[name] = np.nan

        is_blank = result["student_answer"].map(clean_text) == ""
        nonblank = result[~is_blank]
        if len(nonblank) > 0:
            X, _ = self.feature_matrix(nonblank)
            result.loc[~is_blank, "predicted_label"] = self.classifier.predict(X)
            result.loc[~is_blank, FEATURE_NAMES] = X
        return result

    def predict_one(self, question_id, answer):
        """Predict one answer. Returns label, numbers and key point evidence."""
        answer = clean_text(answer)
        if question_id not in self.bank:
            raise ValueError(f"Unknown question_id: {question_id}")
        if answer == "":
            return {"label": "blank", "features": {}, "key_points": []}

        features, evidence = get_features(answer, self.bank[question_id], self.similarity)
        label = self.classifier.predict(np.array([features]))[0]
        return {
            "label": label,
            "features": {n: round(v, 3) for n, v in zip(FEATURE_NAMES, features)},
            "key_points": evidence,
        }

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


# ---------------------------------------------------------------- analysis

def summarize(predictions, bank, by="question_id"):
    """Count labels per question (or per topic) and find the weak ones.

    not_fully_correct_pct = (partial + incorrect) / (correct + partial + incorrect)
    Blank answers are counted separately and are not treated as incorrect.
    """
    data = predictions.copy()
    data["topic"] = data["question_id"].map(lambda q: bank[q]["topic"])

    counts = pd.crosstab(data[by], data["predicted_label"])
    for label in LABELS + ["blank"]:
        if label not in counts.columns:
            counts[label] = 0
    counts = counts[LABELS + ["blank"]]

    graded = counts[LABELS].sum(axis=1)
    counts["graded"] = graded
    counts["not_fully_correct_pct"] = (
        100 * (counts["partial"] + counts["incorrect"]) / graded.replace(0, np.nan)
    ).round(1)
    return counts.reset_index().sort_values("not_fully_correct_pct", ascending=False)


def group_similar_mistakes(predictions, similarity, min_group_size=2, distance_threshold=0.8):
    """Find recurring mistakes: group similar partial/incorrect answers.

    Answers are only grouped with other answers to the SAME question.
    Two answers are close when their cosine similarity is high
    (distance = 1 - similarity). Returns the biggest groups first.
    """
    wrong = predictions[predictions["predicted_label"].isin(["partial", "incorrect"])]
    groups = []

    for qid, rows in wrong.groupby("question_id"):
        answers = [clean_text(a) for a in rows["student_answer"]]
        if len(answers) < 2:
            cluster_ids = [0] * len(answers)
        else:
            vectors = np.array([similarity.embed(a) for a in answers])
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms
            distance = np.clip(1 - vectors @ vectors.T, 0, 2)
            np.fill_diagonal(distance, 0)
            cluster_ids = AgglomerativeClustering(
                n_clusters=None, metric="precomputed", linkage="average",
                distance_threshold=distance_threshold,
            ).fit_predict(distance)

        for cluster in sorted(set(cluster_ids)):
            members = [i for i, c in enumerate(cluster_ids) if c == cluster]
            if len(members) < min_group_size:
                continue
            groups.append({
                "question_id": qid,
                "count": len(members),
                "share_of_wrong_answers_pct": round(100 * len(members) / len(answers), 1),
                "examples": [answers[i] for i in members[:3]],
            })
    return sorted(groups, key=lambda g: g["count"], reverse=True)
