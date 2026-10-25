"""
Part 1 - Data preparation.

Reads the raw GoEmotions TSV files, maps the 28 original emotion labels onto the
six classes this project needs, cleans the text, balances the neutral class and
writes stratified train / validation / test splits.

Run:  python src/data_prep.py
"""
import re
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_label_names():
    """The 28 GoEmotions label names, in the order the ids refer to."""
    return C.RAW.joinpath("emotions.txt").read_text().strip().split("\n")


def load_raw():
    """Concatenate the three official GoEmotions splits into one frame.

    The original splits are re-made later, because our label set and our
    filtering differ from theirs, so their split boundaries no longer apply.
    """
    frames = []
    for part in ("train", "dev", "test"):
        path = C.RAW / f"goemotions_{part}.tsv"
        df = pd.read_csv(path, sep="\t", header=None,
                         names=["text", "label_ids", "comment_id"],
                         dtype={"label_ids": str})
        df["origin"] = part
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
def map_labels(df, label_names):
    """Map each row's GoEmotions labels onto our six classes.

    GoEmotions is multi-label: a comment can carry several emotions. We keep a
    row only when all of its labels that we recognise agree on one target class.
    A comment tagged both 'gratitude' and 'anger' is genuinely ambiguous, and
    training on it would teach the model contradictions.
    """
    def convert(ids):
        names = [label_names[int(i)] for i in str(ids).split(",")]
        mapped = {C.GOEMOTIONS_MAP[n] for n in names if n in C.GOEMOTIONS_MAP}
        return mapped.pop() if len(mapped) == 1 else None

    df = df.copy()
    df["emotion"] = df["label_ids"].map(convert)
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
URL = re.compile(r"https?://\S+|www\.\S+")
USER = re.compile(r"\[NAME\]|\[RELIGION\]|@\w+")
SPACE = re.compile(r"\s+")


def clean_text(s):
    """Light cleaning only.

    Casing and punctuation are deliberately preserved: capitals and exclamation
    marks carry real signal about anger and urgency, and are used as features
    later. Aggressive lowercasing here would throw that away.
    """
    s = URL.sub(" ", str(s))
    s = USER.sub(" ", s)
    s = s.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    return SPACE.sub(" ", s).strip()


def clean_frame(df):
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    df["n_tokens"] = df["text"].str.split().str.len()
    before = len(df)
    df = df[(df["n_tokens"] >= C.MIN_TOKENS) & (df["text"].str.len() <= C.MAX_CHARS)]
    df = df.drop_duplicates(subset="text")
    print(f"  cleaning removed {before - len(df):,} rows "
          f"(too short, too long, or duplicate)")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------
def cap_neutral(df, seed):
    """Stop 'neutral' from swamping the training signal.

    Neutral is by far the largest class in GoEmotions. A classifier trained on
    the raw distribution maximises accuracy by predicting neutral constantly,
    which is useless for triage. Capping it at the size of the largest real
    emotion keeps the data honest without inventing examples.
    """
    counts = df["emotion"].value_counts()
    non_neutral_max = counts.drop("neutral", errors="ignore").max()
    cap = int(non_neutral_max * C.NEUTRAL_CAP_RATIO)
    neutral = df[df["emotion"] == "neutral"]
    if len(neutral) > cap:
        keep = neutral.sample(cap, random_state=seed)
        df = pd.concat([df[df["emotion"] != "neutral"], keep], ignore_index=True)
        print(f"  neutral capped from {len(neutral):,} to {cap:,}")
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def make_splits(df, seed):
    """Stratified train / validation / test split.

    Stratifying keeps the class proportions identical in all three sets, so a
    rare class does not vanish from the test set and make its recall unmeasurable.
    The test set is written once and must not be used for any tuning.
    """
    train_val, test = train_test_split(
        df, test_size=C.TEST_SIZE, stratify=df["emotion"], random_state=seed)
    val_share = C.VAL_SIZE / (1 - C.TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_share, stratify=train_val["emotion"],
        random_state=seed)
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


# ---------------------------------------------------------------------------
def main():
    print("Part 1 - Data preparation")
    print("=" * 58)

    label_names = load_label_names()
    raw = load_raw()
    print(f"\nLoaded {len(raw):,} raw GoEmotions rows across 28 labels")

    mapped = map_labels(raw, label_names)
    kept = mapped[mapped["emotion"].notna()]
    print(f"Mapped onto 6 classes: kept {len(kept):,}, "
          f"dropped {len(mapped) - len(kept):,} "
          f"(unmapped emotion, or conflicting labels)")

    print("\nCleaning:")
    cleaned = clean_frame(kept)

    print("\nClass counts after cleaning:")
    for name, n in cleaned["emotion"].value_counts().items():
        print(f"  {name:<16} {n:>7,}")

    print("\nBalancing:")
    balanced = cap_neutral(cleaned, C.RANDOM_SEED)

    train, val, test = make_splits(balanced, C.RANDOM_SEED)
    cols = ["text", "emotion", "n_tokens"]
    train[cols].to_csv(C.PROCESSED / "train.csv", index=False)
    val[cols].to_csv(C.PROCESSED / "val.csv", index=False)
    test[cols].to_csv(C.PROCESSED / "test.csv", index=False)
    balanced[cols].to_csv(C.PROCESSED / "all.csv", index=False)

    print("\nFinal dataset")
    print("-" * 58)
    print(f"  train  {len(train):>7,}")
    print(f"  val    {len(val):>7,}")
    print(f"  test   {len(test):>7,}")
    print(f"  total  {len(balanced):>7,}")
    print("\nClass distribution (train):")
    dist = train["emotion"].value_counts(normalize=True).sort_index()
    for name, frac in dist.items():
        bar = "#" * int(frac * 60)
        print(f"  {name:<16} {frac*100:>5.1f}%  {bar}")
    print(f"\nWritten to {C.PROCESSED}")


if __name__ == "__main__":
    main()
