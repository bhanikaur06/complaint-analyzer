# Complaint Emotion & Urgency Analyzer — progress

A system that reads a customer complaint and returns the writer's emotion, an urgency score and a triage priority, so a support team knows what to handle first.

---

## Status

| Part | Scope | Status |
|---|---|---|
| 1 | Dataset construction from GoEmotions | **complete** |
| 2 | Emotion classifier, domain transfer and adaptation | **complete** |
| 2.5 | Web interface serving the trained model | **complete** |
| 3 | Accuracy improvement, three new features, deployment | **complete** |
| 3b | Customer details, graded safety scoring, dashboard | **complete** |
| 3c | Login, sign-up and session security | **complete** |
| 4 | Urgency model learned from the P1–P4 labels | next |

---

## To see it running

```
python src/app.py
```

Opens a browser at `http://localhost:8000`. Type a complaint, get a result from the trained model.

---

## What has been built

**A real dataset.** GoEmotions (Demszky et al., ACL 2020, Google Research) — 54,263 human-annotated comments. Mapped from 28 emotion labels down to six triage classes, cleaned, neutral-balanced, split 70/15/15 into **33,075 examples**.

**A second, in-domain dataset.** 286 complaints written and hand-labelled for both emotion and P1–P4 priority, spanning ten sectors and several registers including Hinglish.

**A trained classifier.** TF-IDF over word 1–2 grams and character 3–5 grams, logistic regression with balanced class weights. **0.679 macro-F1**, 5-fold cross-validated.

**A web application.** Python standard-library server, loads the trained model, returns emotion probabilities, urgency score, department routing and a safety override. No framework, nothing extra to install.

**Seven reproducible scripts.** Everything reruns from the command line under a fixed seed.

---

## The result worth presenting

The first model scored **0.597 macro-F1** on the GoEmotions test set. It looked finished.

Run on actual complaints, it produced this:

| Complaint | Predicted | Correct |
|---|---|---|
| "There is a gas leak in my kitchen, please send someone urgently." | neutral | anxiety |
| "This is absolutely unacceptable, you have charged me twice!" | **satisfaction** | anger |

Measured properly on the in-domain set: **0.280 macro-F1**. A transfer gap of 0.299.

The cause was domain. GoEmotions is Reddit comments — a random training row reads *"Dude has a wicked coke nail... just saying."* It had never encountered *debited*, *escalate*, *consumer court* or *gas leak*.

A controlled experiment followed:

| Training data | macro-F1 on complaints |
|---|---|
| 28,113 Reddit comments | 0.280 |
| 229 complaints | **0.679** |
| Both, complaints upweighted 120× | 0.492 |

**229 in-domain examples outperformed 28,113 out-of-domain ones.**

Mixing the two landed between them rather than above both: TF-IDF vocabulary and IDF weights stay dominated by the larger corpus even when the smaller one is repeated 120 times. Naive upweighting does not overcome that.

A second bug surfaced in the same investigation — wrapping the SVM in `CalibratedClassifierCV` to get probabilities made predictions measurably worse, because sigmoid calibration on imbalanced data pulls toward the majority class. Logistic regression replaced it and gives probabilities natively.

---

## Part 3 — Accuracy, features, deployment

### Accuracy: 68.0% → 73.8%

Four ideas tested under identical conditions — 5-fold cross-validation repeated over three shuffles, 15 fits each — so one lucky split can't make an idea look better than it is.

| Idea | Accuracy | Macro-F1 |
|---|---|---|
| Baseline (Part 2 model) | 68.0% | 0.678 |
| + emotion lexicon features | 71.7% | 0.716 |
| + GoEmotions stacking | 70.2% | 0.699 |
| **+ both, regularisation tuned** | **73.8%** | **0.735** |

**Emotion lexicon.** Counts of common emotion words per class, plus writing-style signals: exclamation marks, capitals, question marks, length, negation rate. A positive word preceded by a negation ("not happy") counts toward disappointment.

**GoEmotions stacking.** Part 2 showed that mixing 28,000 Reddit comments into training made results worse. Here the Reddit-trained model is used differently: it makes a prediction, and its six probabilities become six extra features for the complaint model. GoEmotions contributes what it learned about emotion without its vocabulary swamping the complaint features. Its model never sees a complaint label, so nothing leaks.

The two ideas help independently and stack.

### The result that had to be thrown away

The first lexicon scored **80.9%** — a 12.9-point jump.

It was checked before being trusted. The same lexicon added only **0.8 points on GoEmotions**, which it had never been tuned against. The reason: the word list was written by the same person who wrote the complaint corpus, and it contained words like *washout*, *soured* and *insulin* picked after reading those complaints. It had partly memorised the test set.

Rebuilt using only generic words any emotion dictionary would contain, the honest gain is +3.7 points on its own. The reported 73.8% uses that generic list.

### Three new features

**1. Word-level explanation.** The app highlights which words drove the prediction. Each word is removed in turn and the drop in the predicted emotion's probability is measured — leave-one-word-out. Works for any model, unlike reading coefficients, which would mislead with stacked features. Costs about 10 ms.

**2. Needs-review flag.** When confidence is below 60%, the app says so and names the two emotions it is torn between. The threshold was chosen from measured data:

| Confidence | Share of complaints | Accuracy |
|---|---|---|
| below 60% — flagged | 25% | 43.1% |
| 60% and above | 75% | 84.6% |

The flag reliably identifies the predictions most likely to be wrong.

**3. Triage queue.** Analyzed complaints collect in a table sorted by urgency, with resolve buttons and CSV export. Turns single predictions into the actual workflow a support team needs.

### Deployment

`render.yaml` deploys to Render's free tier. Runtime dependencies are pinned in `requirements-deploy.txt` to the exact scikit-learn version the model was saved with. Tested by installing into a clean environment with only those four libraries and confirming the server starts, passes its health check and serves predictions with no version warnings.

### Limitations found during Part 3

- **Overconfidence on some errors.** "My order arrived but one item is missing" is predicted anxiety at 88% when it is closer to disappointment. The review flag catches uncertain mistakes, not confident ones.
- The regularisation strength was tuned on the same cross-validation folds it is reported on, which flatters the final figure slightly. The effect is small: 73.2% at the default setting against 73.8% tuned.

---

## Part 3b — Customer details, graded safety scoring, dashboard

### Customer details

Each complaint added to the queue carries the customer's name, phone, email, customer or order ID and preferred contact method. At least a phone number or an email is required, and the preferred method must have a matching detail. Numbers are checked (7 to 15 digits, international format accepted) and so are email addresses. The queue shows each customer with a clickable call link and email link, and the CSV export includes every field.

- Contact details stay in the browser. The page sends only the complaint text to the model; the test suite confirms this.
- Everything shown in the queue is escaped, so a customer name containing HTML cannot run as code.
- CSV cells that begin with `=`, `+`, `-` or `@` are neutralised, so a name such as `=HYPERLINK(...)` opens as text in Excel instead of executing. Genuine phone numbers like `+919876543210` are left untouched.
- The same complaint cannot be added twice for the same customer, but can for different customers.

### A bug found by the person testing it: every hazard scored exactly 78

The safety rule set every hazard to a flat 78, so a calm gas leak and a house fire looked identical. Checking this exposed a second, worse problem I had introduced the turn before: phrases like "did not make" and bare "burning" were so loose that **"The delivery did not make it on time" scored P1 critical**, as did "My parcel never made it to my house".

The hazard check was rebuilt:

- Each rule is a regular expression that needs context. Fraud, for example, requires a money word (transaction, payment, transfer) near "I never made".
- Two tiers set a floor: tier A, danger to life or property (gas, fire, electrical, medical emergency, trapped, break-in), never below 90; tier B, serious harm if it waits (medicine shortage, unauthorised transaction, account compromised), never below 76.
- Points are added on top for things that make a situation worse: it is happening now, children or elderly people are involved, shouting. The result is the larger of the ordinary score and floor-plus-extras, so the same evidence is never counted twice.
- A tier B hazard is excused when the model reads the message as satisfaction and the text uses resolved-language ("flagged", "handled", "fixed"). Tier A is never excused.

| Check | Result |
|---|---|
| Ordinary complaints wrongly flagged as hazards | 0 of 12 (was 7 of 12) |
| Genuine hazards caught | 13 of 13 |
| Distinct scores among the hazards | 9, from 76 to 100 (was always 78) |

A calm "gas leak, no rush" scores 90; "gas smell, kids at home, send someone now" scores 97; "GAS LEAK!!! my elderly mother is alone, send someone immediately!" scores 100.

The 25 test sentences were written after seeing the failures, so they are regression tests, not an independent measure of accuracy.

### Urgency engine against all 286 labelled complaints

| Measure | Original rules | Current rules |
|---|---|---|
| Labelled P1 complaints caught | 8 of 51 (16%) | 14 of 51 (27%) |
| Precision when it says P1 | 80% | 88% |
| Exact priority match | 36.7% | 38.8% |
| Within one priority band | 71.7% | 73.8% |

It is cautious: when it says P1 it is usually right, but it misses most critical complaints. What it still misses are judgement calls such as harassment, a drunk technician and area-wide outages, which a keyword list cannot capture.

**Emotion is a trained model at 73.8% accuracy; urgency is still hand-written rules at 38.8% exact match.** Learning urgency from the P1-P4 labels is the clearest next step and is Part 4.

### Dashboard

Four live figures (complaints in queue, open P1 critical, average urgency, flagged for human review) and three bar charts (emotions, priorities, departments), updating whenever the queue changes.

### Known errors

- "The sixth time I have called... nobody has fixed it" is predicted anxiety at 84%; frustration is the right label. The model is confidently wrong, which the review flag cannot catch.
- "Your delivery guy was rude and threw the parcel" scores P4 despite 100% anger confidence, because the rules weight only the emotion's strength, not the seriousness of the situation.
- "My account was hacked. Thank you for handling it" stays P1, because "handling" is not in the resolved-language list. This errs on the safe side.

---

## Part 3c — Login and sign-up

A deployed link is public, so the app now requires an account before it will analyze anything.

**What a visitor sees.** Anyone not signed in gets a sign-in / create-account page. Once signed in they get the analyzer, their email and a Log out button. The queue is kept separately for each account in the browser.

### How it is protected

| Threat | Defence |
|---|---|
| Stolen database | Passwords are never stored, only a salted scrypt hash. Session tokens are stored as SHA-256 hashes, so a leaked database cannot be replayed as logins. |
| Password guessing | 5 wrong passwords lock that account for 10 minutes, and 40 failures from one address block it. A locked account is refused before any hashing happens, so guessing costs the attacker and not the server. |
| Finding out who has an account | A wrong email and a wrong password get the identical answer. An unknown email still pays for one hash, so response time does not give it away (measured 137 ms known vs 117 ms unknown). |
| SQL injection | Every query is parameterised. Tested with `' OR 1=1 --` in the email, password and cookie. |
| Cross-site request forgery | The session cookie is `SameSite=Lax`, requests must be JSON, and a POST whose `Origin` differs from the site is refused with 403. |
| Cookie theft by page scripts | The cookie is `HttpOnly`, and `Secure` whenever the request arrived over HTTPS. |
| Memory exhaustion | scrypt uses about 32 MB per hash, so at most two run at once. Twelve simultaneous logins completed in 1.5 s with a peak of two. |
| Slow or oversized requests | Bodies over 20 KB are refused (413), stalled connections are dropped after 15 s, and the server is multi-threaded so one slow client cannot block the others. |
| Clickjacking, content sniffing | `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff` and a Content-Security-Policy that allows only this site. |
| Weak passwords | Minimum 8 characters, a list of common passwords is refused, and so is a password equal to the email name. |
| Information leaks | Server errors return a generic message, the Python version is not announced, and request details are never logged. |
| Contact details leaking between accounts | Account ids are random, so a recreated account never inherits an earlier one's stored queue. Contact details stay in the browser; the model receives only the complaint text (checked by test). |

The account database is written to the user's home folder, not the project folder, so uploading the project to GitHub by drag-and-drop cannot publish it.

### Tests

| Suite | Checks | Result |
|---|---|---|
| Account logic on its own (hashing, rules, lockout, sessions, concurrency) | 42 | all pass |
| Running server, attacked over HTTP (forgery, tampering, oversize, lockout, headers) | 46 | all pass |
| Both pages' JavaScript against a stubbed browser | 21 | all pass |
| Earlier customer-details tests, re-run | 14 | all pass |
| Clean environment with only the four pinned libraries, started as Render starts it | login, analysis, Secure cookie | works |

### Limits to state openly

- **Accounts are temporary on the free hosting plan.** The disk is wiped when the app sleeps or restarts. A demo account created from environment variables is recreated every start, so one login always works. Permanent accounts need a persistent disk or a hosted database, which is not built.
- **No email verification and no password reset.** The app sends no email, so an email address is only a username and a forgotten password cannot be recovered.
- **Each person's queue lives in their own browser.** It does not follow them to another device, and on a shared computer it stays in that browser after logging out.
- **Per-address limits are best effort behind a proxy**, because the address comes from a header. Forging it only avoids the per-address limit; the per-account lock still applies.
- **The lockout can be used to annoy someone**: typing a user's email wrongly five times locks them out for ten minutes.
- **Counters reset when the server restarts.**
- **The Content-Security-Policy still allows inline scripts**, because the page is a single file. It blocks loading anything from other sites, but it is not a defence against injected script on its own; the page escapes all user text for that.
- **The safety rule cannot tell a live hazard from a mention of one.** "The gas leak you fixed last week is sorted", "What is your policy if there is a gas leak?" and "Thank you for dealing with the smoke problem" all score P1. The danger-to-life tier is deliberately not loosened, because a missed gas leak costs far more than a false alarm and no word list can anticipate every phrasing of a leak that is still happening. The proper fix is a trained urgency model with labelled examples of past and hypothetical mentions, which the current corpus does not contain.
- **Tested with scripts, not in a real browser.** Open it yourself before presenting.

---

## Current performance (Part 2)

Emotion, 5-fold cross-validated on complaint text: **0.679 macro-F1**

Spot check, six out of six correct:

| Complaint | Prediction | Confidence |
|---|---|---|
| gas leak in my kitchen | anxiety | 86.1% |
| third time writing, nobody replied | frustration | 55.9% |
| thank you for sorting this out | satisfaction | 90.9% |
| expected much better, very disappointed | disappointment | 89.9% |
| please share the invoice | neutral | 91.8% |
| unacceptable, charged me twice | anger | 95.6% |

---

## Learning curve

| Training examples | macro-F1 |
|---|---|
| 57 | 0.416 |
| 114 | 0.537 |
| 171 | 0.620 |
| 228 | 0.679 |

Still rising. Labelling more complaints is the cheapest remaining improvement — ahead of trying a transformer, ahead of further tuning.

---

## Honest limitations

- The complaint corpus is authored for this project, not sampled from live support tickets.
- 286 examples is small; the ±0.033 standard deviation across folds is real uncertainty.
- One annotator, so there is no inter-annotator agreement figure.
- Frustration is the weakest class and overlaps genuinely with anger and neutral.
- Urgency is still rule-based. Learning it from the priority labels is Part 4.
- English and light Hinglish only.

---

## Files

```
src/config.py            paths, constants, label mapping
src/data_prep.py         Part 1 - clean, map, balance, split
src/explore.py           Part 1 - EDA charts and statistics
src/train_emotion.py     Part 2 - model comparison, tuning, evaluation
src/build_domain_set.py  Part 2 - assemble the complaint corpus
src/domain_adapt.py      Part 2 - transfer experiment, final model
src/predict.py           Part 2 - command-line classifier
src/features.py          Part 3 - lexicon and stacking feature extractors
src/improve.py           Part 3 - accuracy experiments, saves final model
src/app.py               web server: routes, sessions, explanations, review flag, queue
src/auth.py              accounts, password hashing, sessions, lockout
src/login_page.py        the sign-in and create-account page

render.yaml              deployment configuration for Render
requirements-deploy.txt  pinned runtime dependencies
DEPLOY.md                step-by-step deployment guide

reports/PART-1-data.md            dataset construction
reports/PART-2-emotion-model.md   model training and domain adaptation
reports/01-09 *.png               nine charts
models/emotion_final.joblib       the trained model
```

Reproduce everything:

```
python src/data_prep.py
python src/explore.py
python src/train_emotion.py
python src/build_domain_set.py
python src/domain_adapt.py
python src/improve.py
python src/app.py
```
