"""
Part 2b - Domain transfer and adaptation.

A model trained on GoEmotions scores well on GoEmotions. The question this
script answers is whether it works on actual complaints, and if not, what to do
about it.

Three experiments, each evaluated on a held-out slice of the complaint-domain
set so the comparison is fair:

  A  source only      - trained on GoEmotions alone, the Part 2 model
  B  target only      - trained on the small complaint set alone
  C  adapted          - trained on both, with complaint examples upweighted

Run:  python src/domain_adapt.py
"""
import sys
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C

warnings.filterwarnings("ignore")

INK, MUTED, LINE = "#16202B", "#5D6B77", "#C4CCD3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": LINE, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})

# How many times each complaint example is repeated during training.
# 106 examples against 28,000 would otherwise be invisible to the optimiser.
UPWEIGHT = 120


def make_pipeline():
    """Logistic regression over word and character n-grams.

    Logistic regression replaces the linear SVM used in the first pass: it gives
    calibrated probabilities natively, which the API needs, and avoids the
    separate calibration wrapper that measurably degraded predictions.
    """
    return Pipeline([
        ("vec", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                     sublinear_tf=True, strip_accents="unicode")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=3, sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(max_iter=2000, C=4.0,
                                   class_weight="balanced")),
    ])


def evaluate(model, X, y, name):
    pred = model.predict(X)
    return {"setting": name,
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y, pred)}


def main():
    print("Part 2b - Domain transfer and adaptation")
    print("=" * 62)

    src_train = pd.read_csv(C.PROCESSED / "train.csv")
    src_val = pd.read_csv(C.PROCESSED / "val.csv")
    src_test = pd.read_csv(C.PROCESSED / "test.csv")
    tgt = pd.read_csv(C.PROCESSED / "complaints_domain.csv")

    source = pd.concat([src_train, src_val], ignore_index=True)
    print(f"source (GoEmotions)   {len(source):,} train, {len(src_test):,} test")
    print(f"target (complaints)   {len(tgt)} examples, evaluated by 5-fold CV")

    # ---------------------------------------------------------------
    # Experiment A - how far does the source-only model transfer?
    # ---------------------------------------------------------------
    print("\nA. Source-only model")
    print("-" * 62)
    model_a = make_pipeline().fit(source["text"], source["emotion"])
    a_src = evaluate(model_a, src_test["text"], src_test["emotion"], "source test")
    a_tgt = evaluate(model_a, tgt["text"], tgt["emotion"], "complaint set")
    print(f"  GoEmotions test   macro-F1 {a_src['macro_f1']:.3f}   acc {a_src['accuracy']:.3f}")
    print(f"  Complaint set     macro-F1 {a_tgt['macro_f1']:.3f}   acc {a_tgt['accuracy']:.3f}")
    drop = a_src["macro_f1"] - a_tgt["macro_f1"]
    print(f"  Transfer gap      {drop:+.3f} macro-F1")

    # ---------------------------------------------------------------
    # Experiments B and C, scored by cross-validation on the target set.
    # The target set is too small for a single held-out split to mean
    # anything, so every example is used for testing exactly once.
    # ---------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_SEED)
    res = {"A source only": [], "B target only": [], "C adapted": []}

    print("\nB and C. Five-fold cross-validation on the complaint set")
    print("-" * 62)
    for fold, (tr_i, te_i) in enumerate(skf.split(tgt["text"], tgt["emotion"]), 1):
        tr, te = tgt.iloc[tr_i], tgt.iloc[te_i]

        # A: source-only model, same for every fold, scored on this fold's test
        pa = model_a.predict(te["text"])
        res["A source only"].append(f1_score(te["emotion"], pa, average="macro",
                                             zero_division=0))

        # B: trained only on the other folds of the complaint set
        mb = make_pipeline().fit(tr["text"], tr["emotion"])
        pb = mb.predict(te["text"])
        res["B target only"].append(f1_score(te["emotion"], pb, average="macro",
                                             zero_division=0))

        # C: source data plus the complaint folds, repeated UPWEIGHT times
        aug = pd.concat([source[["text", "emotion"]]] +
                        [tr[["text", "emotion"]]] * UPWEIGHT, ignore_index=True)
        mc = make_pipeline().fit(aug["text"], aug["emotion"])
        pc = mc.predict(te["text"])
        res["C adapted"].append(f1_score(te["emotion"], pc, average="macro",
                                         zero_division=0))
        print(f"  fold {fold}   A {res['A source only'][-1]:.3f}   "
              f"B {res['B target only'][-1]:.3f}   C {res['C adapted'][-1]:.3f}")

    print("\nMean macro-F1 on complaint domain")
    print("-" * 62)
    summary = {}
    for k, v in res.items():
        summary[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
        print(f"  {k:<18} {np.mean(v):.3f}  +/- {np.std(v):.3f}")

    # ---------------------------------------------------------------
    # Final model: the winning recipe, fitted on everything available.
    # ---------------------------------------------------------------
    best = max(summary, key=lambda k: summary[k]["mean"])
    print(f"\nBest setting: {best}")

    # The final model uses whichever recipe actually won, not a fixed choice.
    print("\nFitting the final model with the winning recipe...")
    if best.startswith("B"):
        final_data = tgt[["text", "emotion"]]
        recipe = "complaint corpus only"
    else:
        final_data = pd.concat([source[["text", "emotion"]]] +
                               [tgt[["text", "emotion"]]] * UPWEIGHT,
                               ignore_index=True)
        recipe = f"GoEmotions plus complaints upweighted {UPWEIGHT}x"
    final = make_pipeline().fit(final_data["text"], final_data["emotion"])
    print(f"  recipe: {recipe} ({len(final_data):,} rows)")
    print(f"  expected complaint-domain macro-F1: "
          f"{summary[best]['mean']:.3f} (cross-validated, not a re-score)")

    # ---------------------------------------------------------------
    # Does more in-domain data still help? If the curve is rising, the
    # cheapest next improvement is labelling more complaints.
    # ---------------------------------------------------------------
    print("\nLearning curve on the complaint corpus")
    print("-" * 62)
    curve = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        scores = []
        for tr_i, te_i in skf.split(tgt["text"], tgt["emotion"]):
            tr = tgt.iloc[tr_i].sample(frac=frac, random_state=C.RANDOM_SEED)
            if tr["emotion"].nunique() < 2:
                continue
            m = make_pipeline().fit(tr["text"], tr["emotion"])
            scores.append(f1_score(tgt.iloc[te_i]["emotion"],
                                   m.predict(tgt.iloc[te_i]["text"]),
                                   average="macro", zero_division=0))
        n = int(len(tgt) * 0.8 * frac)
        curve.append({"n": n, "macro_f1": float(np.mean(scores))})
        print(f"  {n:>4} examples   macro-F1 {np.mean(scores):.3f}")
    rising = curve[-1]["macro_f1"] - curve[-2]["macro_f1"]
    print("  " + ("Still rising - label more complaints." if rising > 0.015
                  else "Flattening - better features or a stronger model next."))

    joblib.dump(final, C.MODELS / "emotion_final.joblib")

    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    names = list(res.keys())
    means = [summary[k]["mean"] for k in names]
    stds = [summary[k]["std"] for k in names]
    colors = ["#C4CCD3", "#C07414", "#3E7C8C"]
    bars = ax.barh(names[::-1], means[::-1], xerr=stds[::-1],
                   color=colors[::-1], height=0.6,
                   error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
    for b, v in zip(bars, means[::-1]):
        ax.text(v + 0.03, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9, color=MUTED)
    ax.set_xlim(0, max(means) * 1.3)
    ax.set_xlabel("macro-F1 on complaint domain, 5-fold CV")
    ax.set_title("Domain adaptation", loc="left", fontsize=11,
                 fontweight="bold", pad=10)
    ax.grid(axis="x", color=LINE, alpha=0.4, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(C.REPORTS / "08_domain_adaptation.png", dpi=160)
    plt.close(fig)

    out = {"source_only_on_source": a_src, "source_only_on_target": a_tgt,
           "transfer_gap": drop, "cv_results": summary,
           "winning_recipe": best, "learning_curve": curve,
           "upweight": UPWEIGHT}
    (C.REPORTS / "part2b_domain.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved final model to {C.MODELS / 'emotion_final.joblib'}")


if __name__ == "__main__":
    main()
