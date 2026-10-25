"""
Part 1 - Exploratory data analysis.

Produces the charts and statistics that describe the prepared dataset, and
writes them to reports/. Run after data_prep.py.

Run:  python src/explore.py
"""
import sys
import math
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C

PALETTE = {
    "anger": "#B23A2B", "frustration": "#C07414", "disappointment": "#75629E",
    "anxiety": "#3E7C8C", "satisfaction": "#4F7A4A", "neutral": "#6E7C87",
}
INK, MUTED, LINE = "#16202B", "#5D6B77", "#C4CCD3"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": LINE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

STOP = set("""a an the and or but if is am are was were be been being i me my we our you your
he she it they them this that these those to of in on at for with from by as so not no do does
did have has had will would can could should there here what which who when how why about just
my i'm it's don't like get got out up all one more very really too than then them".""".split())


def load():
    return pd.read_csv(C.PROCESSED / "all.csv")


def chart_distribution(df):
    counts = df["emotion"].value_counts().reindex(C.EMOTIONS)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=[PALETTE[e] for e in counts.index[::-1]], height=0.62)
    for b, v in zip(bars, counts.values[::-1]):
        ax.text(v + counts.max() * 0.015, b.get_y() + b.get_height() / 2,
                f"{v:,}", va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, counts.max() * 1.14)
    ax.set_xlabel("examples")
    ax.set_title("Class distribution after mapping and balancing", loc="left",
                 fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", color=LINE, alpha=0.4, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(C.REPORTS / "01_class_distribution.png", dpi=160)
    plt.close(fig)


def chart_lengths(df):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    data = [df[df.emotion == e]["n_tokens"].values for e in C.EMOTIONS]
    bp = ax.boxplot(data, orientation="horizontal", tick_labels=C.EMOTIONS,
                    patch_artist=True,
                    widths=0.55, showfliers=False,
                    medianprops=dict(color=INK, lw=1.4))
    for patch, e in zip(bp["boxes"], C.EMOTIONS):
        patch.set_facecolor(PALETTE[e]); patch.set_alpha(0.35)
        patch.set_edgecolor(PALETTE[e])
    ax.set_xlabel("words per example")
    ax.set_title("Text length by class", loc="left", fontsize=11,
                 fontweight="bold", pad=10)
    ax.grid(axis="x", color=LINE, alpha=0.4, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(C.REPORTS / "02_text_length.png", dpi=160)
    plt.close(fig)


def distinctive_words(df, k=12):
    """Words most associated with each class, by smoothed log-odds ratio.

    Raw frequency would just return common words for every class. The log-odds
    ratio compares a word's rate inside the class against its rate everywhere
    else, which surfaces what actually separates the classes.
    """
    tok = df["text"].str.lower().str.findall(r"[a-z']{3,}")
    df = df.assign(tokens=tok)
    total = Counter()
    per_class = {}
    for e in C.EMOTIONS:
        c = Counter(w for toks in df[df.emotion == e]["tokens"] for w in toks
                    if w not in STOP)
        per_class[e] = c
        total.update(c)
    grand = sum(total.values())

    out = {}
    for e in C.EMOTIONS:
        c = per_class[e]
        n = sum(c.values())
        scores = []
        for w, f in c.items():
            if f < 25:
                continue
            p_in = (f + 1) / (n + 1)
            p_out = (total[w] - f + 1) / (grand - n + 1)
            scores.append((w, math.log(p_in / p_out)))
        out[e] = sorted(scores, key=lambda x: -x[1])[:k]
    return out


def chart_words(words):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, e in zip(axes.ravel(), C.EMOTIONS):
        items = words[e][:8][::-1]
        ax.barh([w for w, _ in items], [s for _, s in items],
                color=PALETTE[e], height=0.6)
        ax.set_title(e, loc="left", fontsize=10, fontweight="bold",
                     color=PALETTE[e])
        ax.tick_params(labelsize=8)
        ax.set_xlabel("log-odds", fontsize=8)
        ax.grid(axis="x", color=LINE, alpha=0.35, lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Most distinctive words per class", x=0.012, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(C.REPORTS / "03_distinctive_words.png", dpi=160)
    plt.close(fig)


def chart_funnel():
    stages = ["Raw GoEmotions", "Mapped to 6 classes", "Cleaned", "Balanced"]
    values = [54263, 38582, 37359, 33075]
    fig, ax = plt.subplots(figsize=(7, 2.9))
    bars = ax.barh(stages[::-1], values[::-1], color=INK, height=0.55, alpha=0.85)
    for b, v in zip(bars, values[::-1]):
        ax.text(v + 900, b.get_y() + b.get_height() / 2, f"{v:,}",
                va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, 60000)
    ax.set_title("How many examples survive each stage", loc="left",
                 fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", color=LINE, alpha=0.4, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(C.REPORTS / "04_pipeline_funnel.png", dpi=160)
    plt.close(fig)


def main():
    df = load()
    print("Part 1 - Exploratory data analysis")
    print("=" * 58)
    print(f"\nExamples        {len(df):,}")
    print(f"Classes         {df['emotion'].nunique()}")
    print(f"Median length   {df['n_tokens'].median():.0f} words")
    print(f"Mean length     {df['n_tokens'].mean():.1f} words")
    print(f"95th percentile {df['n_tokens'].quantile(0.95):.0f} words")
    vocab = len(set(w for t in df['text'].str.lower().str.split() for w in t))
    print(f"Vocabulary      {vocab:,} unique tokens")

    imbalance = df["emotion"].value_counts()
    ratio = imbalance.max() / imbalance.min()
    print(f"\nImbalance ratio {ratio:.1f}x "
          f"({imbalance.idxmax()} vs {imbalance.idxmin()})")

    print("\nLength by class (median words):")
    for e in C.EMOTIONS:
        print(f"  {e:<16} {df[df.emotion==e]['n_tokens'].median():>4.0f}")

    chart_distribution(df)
    chart_lengths(df)
    chart_funnel()
    words = distinctive_words(df)
    chart_words(words)

    print("\nMost distinctive words:")
    for e in C.EMOTIONS:
        print(f"  {e:<16} " + ", ".join(w for w, _ in words[e][:7]))

    print(f"\nCharts written to {C.REPORTS}")


if __name__ == "__main__":
    main()
