# Part 2 — Emotion classifier

**Status: complete.** A trained model classifies complaints into six emotions at **0.679 macro-F1**, cross-validated. Getting there required discovering and fixing a domain transfer failure that the standard evaluation completely hid.

---

## 1. Stage one — training on GoEmotions

Six models were compared on the validation set, in order of increasing capability. Headline metric is **macro-F1**, not accuracy: the classes are imbalanced 15.9 to 1, so a model that ignored anxiety entirely would still post a respectable accuracy.

| Model | macro-F1 | accuracy |
|---|---|---|
| Majority baseline | 0.088 | 0.358 |
| Naive Bayes, words | 0.338 | 0.586 |
| Logistic regression, words | 0.555 | 0.629 |
| Linear SVM, words | 0.554 | 0.635 |
| Logistic regression, words + chars | 0.592 | 0.655 |
| Linear SVM, words + chars | 0.584 | 0.654 |

**Character n-grams were worth more than the choice of classifier.** Adding them gained about 0.04 macro-F1, while swapping SVM for logistic regression moved almost nothing. With a median length of 12 words, word features are sparse; character 3–5 grams catch shared stems across *disappoint / disappointing / disappointed* and survive misspellings.

A grid search over regularisation strength preferred heavy regularisation, `C=0.1`, which is the expected result for high-dimensional sparse text.

**Test set result: 0.597 macro-F1, 0.659 accuracy.**

| Class | precision | recall | F1 | support |
|---|---|---|---|---|
| anger | 0.476 | 0.529 | 0.502 | 306 |
| anxiety | 0.540 | 0.777 | 0.637 | 112 |
| disappointment | 0.541 | 0.592 | 0.565 | 368 |
| frustration | 0.423 | 0.409 | 0.416 | 626 |
| neutral | 0.659 | 0.690 | 0.674 | 1,775 |
| satisfaction | 0.829 | 0.745 | 0.785 | 1,775 |

Two things stand out. **Class weighting worked**: anxiety has the fewest examples by far yet achieves 0.777 recall, because `class_weight="balanced"` makes each anxiety example count more during training. And **frustration is the weakest class at 0.416**, which the confusion matrix explains — 32% of frustration is predicted neutral and 11% anger. Frustration sits between the two and has no sharp boundary with either, in the data or in human judgement.

## 2. The failure the standard evaluation hid

The model scored 0.582 macro-F1 on GoEmotions. Run on actual complaints it produced this:

| Complaint | Predicted | Correct |
|---|---|---|
| "There is a gas leak in my kitchen, please send someone urgently." | neutral | anxiety |
| "This is absolutely unacceptable, you have charged me twice!" | **satisfaction** | anger |
| "This is the third time I am writing and nobody has bothered to reply." | neutral | frustration |

Measured properly on 286 hand-labelled complaints: **0.280 macro-F1, a transfer gap of 0.299.**

The cause is domain. GoEmotions is Reddit comments — a random training row reads *"Dude has a wicked coke nail... just saying."* Complaints are formal, transactional, and use vocabulary the model never saw: *debited, refund, escalate, gas leak, consumer court*. A model can look respectable on its own test set and be unusable in production, and only an in-domain evaluation set reveals it.

A second, smaller bug was found at the same time: wrapping the SVM in `CalibratedClassifierCV` to obtain probabilities **made predictions worse**, 3/6 against 4/6 on spot checks. Sigmoid calibration with 3-fold CV on imbalanced data pulls predictions toward the majority classes. Logistic regression replaced it, giving probabilities natively with no wrapper.

## 3. Stage two — closing the gap

A corpus of **286 complaints** was written and hand-labelled for emotion and priority, spread across ten sectors and several registers including Hinglish and shouty text.

Three settings, all scored by 5-fold cross-validation on the complaint corpus so every example is tested exactly once:

| Setting | Training data | macro-F1 |
|---|---|---|
| A — source only | 28,113 GoEmotions | 0.280 ± 0.034 |
| B — target only | 229 complaints | **0.679 ± 0.033** |
| C — adapted | both, complaints upweighted 120× | 0.492 ± 0.036 |

**229 in-domain examples beat 28,113 out-of-domain ones by 0.40 macro-F1.** That is the central result of Part 2.

Mixing the two (C) landed between them rather than above both, which is informative: the TF-IDF vocabulary and IDF weights are fitted on the combined corpus, so they remain dominated by Reddit vocabulary even when complaint rows are repeated 120 times. Naive upweighting does not overcome that.

## 4. Learning curve

| Training examples | macro-F1 |
|---|---|
| 57 | 0.416 |
| 114 | 0.537 |
| 171 | 0.620 |
| 228 | 0.679 |

Still climbing at the right edge, and gaining roughly 0.06 for the last 57 examples. **Labelling more complaints remains the cheapest available improvement** — ahead of trying a transformer, ahead of further tuning.

## 5. Final model

Trained on the complaint corpus using word 1–2 grams plus character 3–5 grams, logistic regression with balanced class weights. Saved to `models/emotion_final.joblib`.

Spot check, all six correct:

| Complaint | Prediction | Confidence |
|---|---|---|
| gas leak in my kitchen | anxiety | 86.1% |
| third time writing, nobody replied | frustration | 55.9% |
| thank you for sorting this out | satisfaction | 90.9% |
| expected much better, very disappointed | disappointment | 89.9% |
| please share the invoice | neutral | 91.8% |
| unacceptable, charged me twice | anger | 95.6% |

## 6. What GoEmotions was still worth

It is not wasted. It established the pipeline, proved the feature engineering (character n-grams, class weighting, regularisation strength) on data large enough for those conclusions to be reliable, and provided the contrast that made the domain gap measurable at all. Without the source-only number there would be nothing to compare 0.679 against.

## 7. Honest limitations

- The complaint corpus is authored for this project, not sampled from live tickets. It will under-represent the messiness of real support text.
- 286 examples is small. The ±0.033 standard deviation across folds is real uncertainty.
- One annotator labelled it, so there is no inter-annotator agreement figure.
- Frustration remains the hardest class and overlaps genuinely with anger and neutral.
- English and light Hinglish only.

## 8. Reproducing

```bash
python src/train_emotion.py     # stage one, GoEmotions
python src/build_domain_set.py  # assemble the complaint corpus
python src/domain_adapt.py      # transfer experiment and final model
python src/predict.py --demo    # try it
```

---

**Next: Part 3 — urgency.** The complaint corpus already carries P1–P4 priority labels, so urgency can be modelled the same way, then combined with the safety-override rule the prototype already uses.
