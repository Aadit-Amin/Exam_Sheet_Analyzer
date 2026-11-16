# Answer Sheet Analyzer (lab mini project)

Compares a student's short answer with the teacher's model answer and key points, suggests **correct / partial / incorrect**, and shows which questions and topics students find hardest and which wrong answers keep recurring.

## Run

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # Windows: .venv\Scripts\python.exe
.venv/bin/python train.py                                # trains + prints results
.venv/bin/python -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000 . If `models/tfidf.joblib` is missing, the server trains it automatically.
Tests: `python -m pytest -q`

## Files

| File | Job |
|---|---|
| `core.py` | cleaning, similarity, features, grader, per-question summary |
| `train.py` | 5-fold grouped cross-validation, baselines, saves the model |
| `main.py` | FastAPI: `/api/questions`, `/api/predict`, `/api/analyze` |
| `static/index.html` | plain HTML/JS page |
| `data/` | 5 networking questions, 95 sample answers, key-point rubric |

## What the analysis shows

- **Commonly missed questions:** per question, the share of graded answers that are partial or incorrect (blank answers counted separately).
- **Topics where students struggle:** the same table grouped by topic.
- **Recurring mistakes:** partial/incorrect answers with similar wording are grouped within each question, with examples and the share of wrong answers each group covers.

## Method

1. Similarity (TF-IDF cosine; optional Sentence-BERT) between answer and model answer.
2. Four numbers per answer: similarity to model answer, mean and minimum best-match score over the key points, and length ratio.
3. Logistic regression turns the four numbers into a label. Blank answers are never graded.
4. Answers sharing a `variant_group` (reworded copies) always stay in the same cross-validation fold.

## Results (TF-IDF, 90 non-blank answers, 5-fold grouped CV)

| Method | Macro-F1 |
|---|---|
| Always predict most common label | 0.167 |
| Similarity thresholds only | 0.418 |
| Logistic regression on 4 features | 0.535 |

Correct answers are found well (recall 0.90). Incorrect and partial answers are often confused, and 9 of 30 incorrect answers were labelled correct.

## Limitations

- The 95 answers are AI-generated and their labels were not checked by a teacher. Results do not show accuracy on real students.
- Only 5 questions, so there is no evidence the model works on new questions.
- TF-IDF matches words, not meaning, so a correct answer in different words can score low.
- Output is a suggestion for a teacher to review, not a grade.
- Running `/api/analyze` on `data/responses.csv` uses answers the model was trained on, so those counts look better than real use would.
- Recurring-mistake groups are based on similar wording. TF-IDF cannot tell "DNS translates names to IP addresses" from "DNS does not translate names to IP addresses", so read the examples before drawing conclusions.
- A per-key-point "which concept was missed" report is not included: with TF-IDF the key-point scores are too low and too similar across labels to be trusted.
- Sentence-BERT (`python train.py --backend sbert`) is included but was not run here.
