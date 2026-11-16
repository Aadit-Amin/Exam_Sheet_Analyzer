"""Train and evaluate the grader.

    python train.py                  # TF-IDF (no downloads)
    python train.py --backend sbert  # needs sentence-transformers + internet once

Evaluation uses 5-fold cross-validation. Answers that share a
variant_group (reworded copies of the same idea) always stay in the same
fold, so the model is never tested on a near-copy of something it trained on.
"""
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from core import AnswerGrader, LABELS, build_bank, load_data


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)


def threshold_predict(similarity, low, high):
    """Baseline: below low = incorrect, below high = partial, else correct."""
    return np.where(similarity < low, "incorrect",
                    np.where(similarity < high, "partial", "correct"))


def best_thresholds(similarity, labels):
    """Try many (low, high) pairs on the training data and keep the best."""
    best = (0.0, 0.0, -1.0)
    steps = np.linspace(0, 1, 41)
    for low in steps:
        for high in steps:
            if low < high:
                score = macro_f1(labels, threshold_predict(similarity, low, high))
                if score > best[2]:
                    best = (low, high, score)
    return best[0], best[1]


def cross_validate(data, bank, backend):
    """Return a table of true labels and the predictions of each method."""
    folds = GroupKFold(n_splits=5)
    pieces = []

    for train_index, test_index in folds.split(data, groups=data["variant_group"]):
        train = data.iloc[train_index]
        test = data.iloc[test_index]

        grader = AnswerGrader(backend).fit(train, bank)
        X_train, _ = grader.feature_matrix(train)
        X_test, _ = grader.feature_matrix(test)

        # Baseline 1: always predict the most common training label.
        most_common = Counter(train["label"]).most_common(1)[0][0]
        # Baseline 2: thresholds on similarity to the model answer.
        low, high = best_thresholds(X_train[:, 0], train["label"])

        piece = test[["response_id", "question_id", "student_answer", "label"]].copy()
        piece["majority"] = most_common
        piece["similarity_threshold"] = threshold_predict(X_test[:, 0], low, high)
        piece["classifier"] = grader.classifier.predict(X_test)
        piece["reference_similarity"] = X_test[:, 0]
        pieces.append(piece)

    return pd.concat(pieces).sort_values("response_id")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["tfidf", "sbert"], default="tfidf")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    questions, responses, rubrics = load_data(args.data_dir)
    bank = build_bank(questions, rubrics)
    data = responses[responses["label"] != "blank"].reset_index(drop=True)
    print(f"{len(data)} labelled non-blank answers, backend = {args.backend}")

    results = cross_validate(data, bank, args.backend)

    print("\nMacro-F1 (5-fold grouped cross-validation)")
    scores = {}
    for method in ["majority", "similarity_threshold", "classifier"]:
        scores[method] = macro_f1(results["label"], results[method])
        print(f"  {method:22s} {scores[method]:.3f}")

    print("\nClassifier report")
    print(classification_report(results["label"], results["classifier"],
                                labels=LABELS, zero_division=0))
    matrix = confusion_matrix(results["label"], results["classifier"], labels=LABELS)
    print("Confusion matrix (rows = true, columns = predicted)")
    print(pd.DataFrame(matrix, index=LABELS, columns=LABELS))

    # Train the final model on all labelled answers and save it.
    final_grader = AnswerGrader(args.backend).fit(data, bank)
    final_grader.save(f"models/{args.backend}.joblib")

    results.to_csv(f"results/cv_predictions_{args.backend}.csv", index=False)
    with open(f"results/cv_scores_{args.backend}.json", "w") as f:
        json.dump({"backend": args.backend, "macro_f1": scores,
                   "confusion_matrix": matrix.tolist(), "labels": LABELS}, f, indent=2)
    print(f"\nSaved models/{args.backend}.joblib and results/")


if __name__ == "__main__":
    main()
