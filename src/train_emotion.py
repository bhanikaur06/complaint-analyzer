"""
Part 2 - Emotion classifier.

Trains a sequence of models of increasing capability, selects between them on
the validation set, tunes the winner, then evaluates once on the untouched test
set and saves the fitted pipeline.

Headline metric is macro-F1, not accuracy. The classes are imbalanced 15.9 to 1,
so a model that ignored anxiety entirely would still score well on accuracy.
Macro-F1 averages across classes, so every class counts the same regardless of
how many examples it has.

Run:  python src/train_emotion.py
"""
import sys
import time
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (f1_score, accuracy_score, classification_report,
                             confusion_matrix)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C

warnings.filterwarnings("ignore", category=UserWarning)

PALETTE = {"anger": "#B23A2B", "frustration": "#C07414",
           "disappointment": "#75629E", "anxiety": "#3E7C8C",
           "satisfaction": "#4F7A4A", "neutral": "#6E7C87"}
INK, MUTED, LINE = "#16202B", "#5D6B77", "#C4CCD3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": LINE, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})


# ---------------------------------------------------------------------------
def load_splits():
    tr = pd.read_csv(C.PROCESSED / "train.csv")
    va = pd.read_csv(C.PROCESSED / "val.csv")
    te = pd.read_csv(C.PROCESSED / "test.csv")
    return tr, va, te


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------
def word_features():
    """Word 1-2 grams.

    sublinear_tf dampens repeated words: saying "terrible" five times is not
    five times as angry. min_df drops words appearing in fewer than 3 documents,
    which removes typos and names that cannot generalise.
    """
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                           strip_accents="unicode", lowercase=True)


def char_features():
    """Character 3-5 grams.

    Examples average 12 words, so word features are sparse. Character n-grams
    catch morphology ("disappoint", "disappointing", "disappointed" share stems)
    and survive misspellings, which word features cannot.
    """
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           sublinear_tf=True, lowercase=True)


def union_features():
    return FeatureUnion([("word", word_features()), ("char", char_features())])


# ---------------------------------------------------------------------------
def candidates():
    """Models in order of increasing capability, so each step's gain is visible."""
    return [
        ("Majority baseline",
         Pipeline([("vec", word_features()),
                   ("clf", DummyClassifier(strategy="most_frequent"))])),

        ("Naive Bayes, words",
         Pipeline([("vec", word_features()),
                   ("clf", MultinomialNB(alpha=0.3))])),

        ("Logistic regression, words",
         Pipeline([("vec", word_features()),
                   ("clf", LogisticRegression(max_iter=1500, C=4.0,
                                              class_weight="balanced"))])),

        ("Linear SVM, words",
         Pipeline([("vec", word_features()),
                   ("clf", LinearSVC(C=0.5, class_weight="balanced"))])),

        ("Logistic regression, words + chars",
         Pipeline([("vec", union_features()),
                   ("clf", LogisticRegression(max_iter=1500, C=4.0,
                                              class_weight="balanced"))])),

        ("Linear SVM, words + chars",
         Pipeline([("vec", union_features()),
                   ("clf", LinearSVC(C=0.5, class_weight="balanced"))])),
    ]


def compare(train, val):
    rows = []
    for name, pipe in candidates():
        t0 = time.time()
        pipe.fit(train["text"], train["emotion"])
        pred = pipe.predict(val["text"])
        rows.append({
            "model": name,
            "macro_f1": f1_score(val["emotion"], pred, average="macro"),
            "accuracy": accuracy_score(val["emotion"], pred),
            "seconds": time.time() - t0,
            "pipe": pipe,
        })
        print(f"  {name:<36} macro-F1 {rows[-1]['macro_f1']:.3f}   "
              f"acc {rows[-1]['accuracy']:.3f}   {rows[-1]['seconds']:.1f}s")
    return rows


# ---------------------------------------------------------------------------
def tune(train, val):
    """Small grid over the regularisation strength of the winning family.

    Train and validation are concatenated with an explicit fold index so the
    search trains on train and scores on val, never mixing the two.
    """
    from sklearn.model_selection import PredefinedSplit
    X = pd.concat([train["text"], val["text"]], ignore_index=True)
    y = pd.concat([train["emotion"], val["emotion"]], ignore_index=True)
    fold = np.r_[np.full(len(train), -1), np.zeros(len(val))]

    pipe = Pipeline([("vec", union_features()),
                     ("clf", LinearSVC(class_weight="balanced"))])
    grid = {"clf__C": [0.1, 0.25, 0.5, 1.0, 2.0],
            "vec__word__min_df": [2, 3]}
    gs = GridSearchCV(pipe, grid, scoring="f1_macro",
                      cv=PredefinedSplit(fold), n_jobs=-1, verbose=0)
    gs.fit(X, y)
    for p, s in sorted(zip(gs.cv_results_["params"],
                           gs.cv_results_["mean_test_score"]),
                       key=lambda x: -x[1]):
        print(f"  C={p['clf__C']:<5} min_df={p['vec__word__min_df']}   "
              f"macro-F1 {s:.3f}")
    return gs.best_estimator_, gs.best_params_, gs.best_score_


# ---------------------------------------------------------------------------
def plot_comparison(rows, tuned_f1):
    names = [r["model"] for r in rows] + ["Tuned linear SVM"]
    vals = [r["macro_f1"] for r in rows] + [tuned_f1]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    colors = [MUTED] * len(rows) + ["#3E7C8C"]
    colors[0] = LINE
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.62)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, max(vals) * 1.2)
    ax.set_xlabel("macro-F1 on validation set")
    ax.set_title("Model comparison", loc="left", fontsize=11,
                 fontweight="bold", pad=10)
    ax.grid(axis="x", color=LINE, alpha=0.4, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(C.REPORTS / "05_model_comparison.png", dpi=160)
    plt.close(fig)


def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i,j]*100:.0f}", ha="center", va="center",
                    fontsize=8.5,
                    color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix, test set (row %)", loc="left",
                 fontsize=11, fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.78, label="share of true class")
    fig.tight_layout(); fig.savefig(C.REPORTS / "06_confusion_matrix.png", dpi=160)
    plt.close(fig)


def plot_per_class(report, labels):
    f1 = [report[l]["f1-score"] for l in labels]
    sup = [report[l]["support"] for l in labels]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    bars = ax.bar(labels, f1, color=[PALETTE[l] for l in labels], width=0.6)
    for b, v, s in zip(bars, f1, sup):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=8.5, color=MUTED)
        ax.text(b.get_x() + b.get_width() / 2, 0.03, f"n={s}",
                ha="center", fontsize=7.5, color="white")
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 on test set", loc="left", fontsize=11,
                 fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", color=LINE, alpha=0.4, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(C.REPORTS / "07_per_class_f1.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    print("Part 2 - Emotion classifier")
    print("=" * 62)
    train, val, test = load_splits()
    print(f"train {len(train):,}   val {len(val):,}   test {len(test):,}")

    print("\nComparing candidate models on the validation set")
    print("-" * 62)
    rows = compare(train, val)

    print("\nTuning the best family")
    print("-" * 62)
    best, params, best_f1 = tune(train, val)
    print(f"\nBest parameters: {params}")

    print("\nFinal evaluation on the held-out test set")
    print("-" * 62)
    pred = best.predict(test["text"])
    labels = sorted(test["emotion"].unique())
    macro = f1_score(test["emotion"], pred, average="macro")
    acc = accuracy_score(test["emotion"], pred)
    print(f"  macro-F1  {macro:.3f}")
    print(f"  accuracy  {acc:.3f}")
    print()
    print(classification_report(test["emotion"], pred, digits=3))

    report = classification_report(test["emotion"], pred, output_dict=True)
    plot_comparison(rows, best_f1)
    plot_confusion(test["emotion"], pred, labels)
    plot_per_class(report, labels)

    # Calibrated copy so the API can return probabilities, not just a label.
    print("Fitting a calibrated copy for probability outputs...")
    calib = CalibratedClassifierCV(best, method="sigmoid", cv=3)
    calib.fit(pd.concat([train["text"], val["text"]], ignore_index=True),
              pd.concat([train["emotion"], val["emotion"]], ignore_index=True))

    joblib.dump(best, C.MODELS / "emotion_svm.joblib")
    joblib.dump(calib, C.MODELS / "emotion_calibrated.joblib")
    metrics = {"test_macro_f1": macro, "test_accuracy": acc,
               "best_params": {k: str(v) for k, v in params.items()},
               "per_class": {l: report[l] for l in labels},
               "validation_comparison": [
                   {"model": r["model"], "macro_f1": r["macro_f1"],
                    "accuracy": r["accuracy"]} for r in rows]}
    (C.REPORTS / "part2_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved models to {C.MODELS}")
    print(f"Saved charts and metrics to {C.REPORTS}")


if __name__ == "__main__":
    main()
