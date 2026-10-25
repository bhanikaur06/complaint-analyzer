"""
Part 2b - Complaint-domain evaluation set.

GoEmotions is Reddit comments. The system has to work on support tickets, which
use different vocabulary and register. A model can score well on GoEmotions and
still fail in production, so a separate in-domain set is needed to detect that.

This script extracts the 100 hand-labelled complaints embedded in the prototype
(demo-app-v1.0.html) and writes them as a CSV. They are written in the target
domain and labelled by hand against the same six classes.

Run:  python src/build_domain_set.py
"""
import re
import sys
import json
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C

SOURCE = C.ROOT / "demo-app-v1.0.html"


def extract():
    html = SOURCE.read_text(encoding="utf-8")
    start = html.index("const BASE_CORPUS = [")
    end = html.index("].map(", start)
    body = html[start + len("const BASE_CORPUS = ["):end]

    rows = []
    for m in re.finditer(r'\["((?:[^"\\]|\\.)*)","(\w+)","(P\d)"\]', body):
        text, emotion, priority = m.group(1), m.group(2), m.group(3)
        rows.append({"text": text.replace('\\"', '"'),
                     "emotion": emotion, "priority": priority})
    return pd.DataFrame(rows)


def main():
    if not SOURCE.exists():
        sys.exit(f"Could not find {SOURCE}")
    df = extract()
    df["n_tokens"] = df["text"].str.split().str.len()
    out = C.PROCESSED / "complaints_domain.csv"
    df.to_csv(out, index=False)

    print("Part 2b - Complaint-domain set")
    print("=" * 52)
    print(f"Extracted {len(df)} hand-labelled complaints")
    print(f"Median length {df['n_tokens'].median():.0f} words "
          f"(GoEmotions median is 12)")
    print("\nEmotion distribution:")
    for name, n in df["emotion"].value_counts().items():
        print(f"  {name:<16} {n:>3}")
    print("\nPriority distribution:")
    for name, n in df["priority"].value_counts().sort_index().items():
        print(f"  {name:<16} {n:>3}")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
