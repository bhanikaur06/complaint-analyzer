# Complaint Emotion & Urgency Analyzer

An ML system that reads a customer complaint and returns the writer's emotion,
an urgency score and a triage priority, so a support team knows what to handle first.

## Status

| Part | Scope | Status |
|---|---|---|
| 1 | Dataset construction from GoEmotions | complete |
| 2 | Emotion classifier, domain transfer and adaptation | complete |
| 3 | Accuracy improvement, explanations, review flag, queue, deployment | complete |
| 4 | Urgency model from the P1-P4 priority labels | next |
| 4 | Flask API and web interface | planned |
| 5 | Explainability and batch processing | planned |

## Dataset

33,075 human-annotated examples across six classes, derived from GoEmotions
(Demszky et al., ACL 2020). See `reports/PART-1-data.md` for the full method.

## Layout

```
data/raw/          GoEmotions TSV files as downloaded
data/processed/    train.csv, val.csv, test.csv, all.csv
src/config.py      all paths, constants and the label mapping
src/data_prep.py        Part 1 - cleaning, mapping, balancing, splitting
src/explore.py          Part 1 - EDA charts and statistics
src/train_emotion.py    Part 2 - model comparison, tuning, evaluation
src/build_domain_set.py Part 2 - assemble the complaint corpus
src/domain_adapt.py     Part 2 - transfer experiment, final model
src/predict.py          Part 2 - classify text with the trained model
reports/           charts and the written report for each part
models/            serialised model artifacts
```

## Setup

```bash
pip install -r requirements.txt
python src/data_prep.py
python src/explore.py
python src/train_emotion.py
python src/build_domain_set.py
python src/domain_adapt.py
python src/predict.py --demo
```

Deterministic under the seed in `src/config.py`.

## Result so far

Emotion classification on complaint text: **73.8% accuracy, 0.735 macro-F1**,
5-fold cross-validated, repeated over three shuffles. See `reports/PART-2-emotion-model.md` for the method and for
the domain-transfer failure that had to be fixed to get there.

## Signing in

The web app asks for an account. Run `python src/app.py`, then use **Create account** on the page.
Accounts are stored in a `.complaint-analyzer` folder in your home directory, outside the project,
so they are never uploaded with the code. To skip signing up on a hosted copy, set `DEMO_EMAIL` and
`DEMO_PASSWORD` before starting. See `DEPLOY.md`.
