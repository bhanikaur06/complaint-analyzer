"""
Part 3 - Accuracy improvement experiments.

Tests several ideas for improving the complaint emotion classifier against the
current model, all under the identical 5-fold cross-validation, so every number
is directly comparable. Only ideas that beat the baseline are kept.

Ideas tested:
  1. Baseline        - current model, word + char TF-IDF, logistic regression
  2. Lexicon         - adds counts from a hand-built emotion word list
  3. Stacking        - adds the GoEmotions model's probabilities as features
  4. Both            - lexicon and stacking together
  5. Tuned           - the best of the above with its regularisation re-tuned

Run:  python src/improve.py
"""
import re
import sys
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C
from features import LexiconFeatures, SourceModelFeatures

warnings.filterwarnings("ignore")

INK, MUTED, LINE = "#16202B", "#5D6B77", "#C4CCD3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": LINE, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})


def tfidf():
    return [
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                 sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, sublinear_tf=True)),
    ]


def build(lexicon=False, source=None, c=4.0):
    parts = tfidf()
    if lexicon:
        parts.append(("lex", Pipeline([("f", LexiconFeatures()),
                                       ("s", StandardScaler())])))
    if source is not None:
        parts.append(("src", Pipeline([("f", SourceModelFeatures(source)),
                                       ("s", StandardScaler())])))
    return Pipeline([
        ("vec", FeatureUnion(parts)),
        ("clf", LogisticRegression(max_iter=3000, C=c,
                                   class_weight="balanced")),
    ])


def cv_score(make, data, seeds=(42, 7, 2024)):
    """5-fold CV repeated over three shuffles, so a single lucky split
    cannot make one idea look better than it is."""
    accs, f1s = [], []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in skf.split(data["text"], data["emotion"]):
            m = make().fit(data["text"].iloc[tr], data["emotion"].iloc[tr])
            p = m.predict(data["text"].iloc[te])
            accs.append(accuracy_score(data["emotion"].iloc[te], p))
            f1s.append(f1_score(data["emotion"].iloc[te], p,
                                average="macro", zero_division=0))
    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)


def main():
    print("Part 3 - Accuracy improvement experiments")
    print("=" * 66)
    data = pd.read_csv(C.PROCESSED / "complaints_domain.csv")
    print(f"{len(data)} complaints, 5-fold CV repeated over 3 shuffles "
          f"(15 fits per idea)\n")

    # The GoEmotions model is trained once on Reddit data only. It never sees
    # a complaint, so using its outputs as features cannot leak test labels.
    print("Training the GoEmotions source model for stacking...")
    src = pd.concat([pd.read_csv(C.PROCESSED / "train.csv"),
                     pd.read_csv(C.PROCESSED / "val.csv")])
    source = Pipeline([("vec", FeatureUnion(tfidf())),
                       ("clf", LogisticRegression(max_iter=2000, C=4.0,
                                                  class_weight="balanced"))])
    source.fit(src["text"], src["emotion"])
    # The source model is embedded inside the final model, no separate file.

    ideas = [
        ("1  Baseline (current)", lambda: build()),
        ("2  + emotion lexicon", lambda: build(lexicon=True)),
        ("3  + GoEmotions stacking", lambda: build(source=source)),
        ("4  + lexicon + stacking", lambda: build(lexicon=True, source=source)),
    ]

    print("\n" + f"{'Idea':<30}{'Accuracy':>16}{'Macro-F1':>16}")
    print("-" * 66)
    results = []
    for name, make in ideas:
        a, a_sd, f, f_sd = cv_score(make, data)
        results.append({"idea": name, "acc": a, "acc_sd": a_sd,
                        "f1": f, "f1_sd": f_sd})
        print(f"{name:<30}{a*100:>9.1f}% ±{a_sd*100:>4.1f}"
              f"{f:>10.3f} ±{f_sd:.3f}")

    # Re-tune regularisation for the winner only.
    best = max(results, key=lambda r: r["f1"])
    use_lex = "lexicon" in best["idea"]
    use_src = "stacking" in best["idea"]
    print(f"\nTuning regularisation for: {best['idea'].strip()}")
    tuned = []
    for c in (1.0, 2.0, 4.0, 8.0, 16.0):
        a, a_sd, f, f_sd = cv_score(
            lambda c=c: build(lexicon=use_lex,
                              source=source if use_src else None, c=c), data)
        tuned.append((c, a, a_sd, f, f_sd))
        print(f"  C={c:<5}{a*100:>8.1f}%{f:>10.3f}")
    c_best, a, a_sd, f, f_sd = max(tuned, key=lambda t: t[3])
    results.append({"idea": f"5  Tuned (C={c_best})", "acc": a, "acc_sd": a_sd,
                    "f1": f, "f1_sd": f_sd})

    base = results[0]
    final = results[-1]
    print("\n" + "=" * 66)
    print(f"Baseline   accuracy {base['acc']*100:.1f}%   macro-F1 {base['f1']:.3f}")
    print(f"Final      accuracy {final['acc']*100:.1f}%   macro-F1 {final['f1']:.3f}")
    print(f"Gain       {(final['acc']-base['acc'])*100:+.1f} points   "
          f"{final['f1']-base['f1']:+.3f}")

    # Fit and save the final model on all complaints.
    model = build(lexicon=use_lex, source=source if use_src else None, c=c_best)
    model.fit(data["text"], data["emotion"])
    joblib.dump(model, C.MODELS / "emotion_final.joblib")
    print(f"\nSaved improved model to {C.MODELS / 'emotion_final.joblib'}")

    # Chart
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    names = [r["idea"][3:] for r in results]
    vals = [r["acc"] * 100 for r in results]
    errs = [r["acc_sd"] * 100 for r in results]
    colors = [LINE] + [MUTED] * (len(results) - 2) + ["#3E7C8C"]
    bars = ax.barh(names[::-1], vals[::-1], xerr=errs[::-1], color=colors[::-1],
                   height=0.6, error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 2.2, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=9, color=MUTED)
    ax.set_xlim(0, 100)
    ax.set_xlabel("accuracy on complaints, 5-fold CV x3")
    ax.set_title("Accuracy improvement experiments", loc="left",
                 fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", color=LINE, alpha=0.4, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(C.REPORTS / "09_accuracy_improvement.png", dpi=160)
    plt.close(fig)

    (C.REPORTS / "part3_improvement.json").write_text(json.dumps(
        {"results": results, "best_C": c_best,
         "uses_lexicon": use_lex, "uses_stacking": use_src}, indent=2))


if __name__ == "__main__":
    main()
