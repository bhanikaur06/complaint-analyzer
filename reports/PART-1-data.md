# Part 1 — Dataset construction

**Status: complete.** A real, human-annotated dataset of 33,075 examples is prepared, cleaned, balanced and split, with the pipeline fully reproducible from a single command.

---

## 1. Source dataset

**GoEmotions** (Demszky et al., *GoEmotions: A Dataset of Fine-Grained Emotions*, ACL 2020, Google Research).

| | |
|---|---|
| Size | 54,263 comments |
| Labels | 27 emotions plus neutral |
| Annotation | Human raters, multi-label |
| Licence | Apache 2.0, freely redistributable |

This was chosen over the alternatives for concrete reasons. Scraped review datasets carry star ratings, not emotions, so a 1-star review labelled "negative" tells us nothing about whether the customer is angry or anxious — a distinction this project depends on. Lexicon-labelled datasets such as those built with VADER are circular: training a model on labels produced by a keyword tool teaches it to imitate the keyword tool. GoEmotions is labelled by people, is large enough to train on, and distinguishes emotions at the granularity the triage problem needs.

The known trade-off is domain: GoEmotions is Reddit comments, not support tickets. This is addressed in Part 3 with a held-out complaint-domain evaluation set.

## 2. Label mapping

The 28 source labels collapse onto the six classes the triage problem requires:

| Target class | Source emotions |
|---|---|
| anger | anger, disgust |
| frustration | annoyance, disapproval |
| disappointment | disappointment, sadness, grief |
| anxiety | fear, nervousness |
| satisfaction | gratitude, admiration, approval, joy, relief |
| neutral | neutral |

Thirteen source emotions with no clear meaning for complaint handling — amusement, curiosity, love, pride, surprise and others — were **dropped rather than forced into a bucket**. A wrong mapping injects noise into every subsequent experiment, which is worse than having less data.

GoEmotions is multi-label, so a comment can carry several emotions at once. A row is kept only when all of its recognised labels agree on one target class. A comment tagged both *gratitude* and *anger* is genuinely ambiguous and would teach the model contradictions.

## 3. Pipeline

| Stage | Remaining | Removed |
|---|---|---|
| Raw GoEmotions | 54,263 | — |
| Mapped to 6 classes | 38,582 | 15,681 unmapped or conflicting |
| Cleaned | 37,359 | 1,223 short, long or duplicate |
| Neutral capped | 33,075 | 4,284 surplus neutral |

**Cleaning** strips URLs, anonymisation placeholders and HTML entities, drops examples under 3 words or over 1,000 characters, and removes exact duplicates.

Casing and punctuation are **deliberately preserved**. Capitals and exclamation marks carry genuine signal about anger and urgency and are used as features later. Routine lowercasing at this stage would discard that.

**Balancing.** Neutral was 43% of the mapped data. Left alone, a classifier maximises accuracy by answering "neutral" to everything, which is useless for triage. Neutral is capped at the size of the largest real emotion class. No examples were duplicated or synthesised.

## 4. Splits

Stratified 70 / 15 / 15, seed 42.

| Split | Examples |
|---|---|
| train | 23,151 |
| validation | 4,962 |
| test | 4,962 |

Stratification keeps class proportions identical across all three, so anxiety does not vanish from the test set and make its recall unmeasurable. The test set is written once and is not touched until final evaluation.

## 5. What the data looks like

| Statistic | Value |
|---|---|
| Median length | 12 words |
| Mean length | 12.9 words |
| 95th percentile | 24 words |
| Vocabulary | 41,802 unique tokens |
| Imbalance ratio | 15.9× (satisfaction vs anxiety) |

Charts in this folder:

1. `01_class_distribution.png` — final class counts
2. `02_text_length.png` — length distribution per class
3. `03_distinctive_words.png` — most characteristic words per class
4. `04_pipeline_funnel.png` — examples surviving each stage

## 6. Findings that shape Part 2

**The label mapping validates.** The most distinctive words per class are exactly what the categories should produce — *disgusting, dare, angry, awful, hate* for anger; *scary, scared, afraid, terrifying, worried* for anxiety; *thank, thanks, appreciate, awesome* for satisfaction. These were derived from the data, not specified, so they are independent evidence the mapping is sound.

**Class imbalance is the main obstacle.** Anxiety holds only 745 examples against satisfaction's 11,834, a ratio of 15.9 to 1. Anxiety is also the class a triage system can least afford to miss, since it covers safety and financial-risk complaints. Part 2 must use class weighting and report macro-F1 rather than plain accuracy, which would otherwise look good while ignoring the rare class entirely.

**Examples are short.** A median of 12 words means each one carries few features. Character n-grams alongside word n-grams are likely to help.

**Neutral is weakly characterised.** Its distinctive words — *send, etc, wall, paid* — are topic words, not emotional ones, which is expected: neutral is defined by the absence of emotion rather than the presence of anything. It will probably be the class most often confused with others.

## 7. Reproducing

```bash
python src/data_prep.py    # builds data/processed/{train,val,test}.csv
python src/explore.py      # writes the charts and statistics above
```

Both are deterministic under seed 42 in `src/config.py`.

---

**Next: Part 2 — train and evaluate the emotion classifier.** Baseline through to tuned model, with class weighting for the imbalance, macro-F1 as the headline metric, and a confusion matrix showing where the errors land.
