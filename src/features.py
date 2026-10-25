"""
Custom feature extractors used by the improved emotion model.

Kept in their own module because a saved model refers to these classes by
import path. Anything that loads emotion_final.joblib must be able to import
this file, which is why app.py adds src/ to the path before loading.
"""
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Generic emotion word lists, dictionary-style.
#
# Deliberately restricted to common emotion vocabulary. An earlier, longer list
# scored 80.9% on the complaint corpus, but it contained words chosen after
# reading those complaints, so it partly memorised the test set: the same list
# added only 0.8 points on GoEmotions, which it had never been tuned against.
# Keeping only words any emotion dictionary would contain gives a smaller but
# honest gain. Matched as word prefixes, so "frustrat" catches frustrated,
# frustrating and frustration.
LEXICON = {
    "anger": ["unacceptable", "outrageous", "disgust", "pathetic", "furious",
              "ridiculous", "angry", "hate", "terrible", "worst", "rude",
              "useless"],
    "frustration": ["again", "still", "keep", "nobody", "no one", "waiting",
                    "frustrat", "annoy", "repeat", "tired of"],
    "disappointment": ["disappoint", "expected", "let down", "regret",
                       "unhappy", "poor", "sad", "worse"],
    "anxiety": ["worried", "worry", "scared", "afraid", "anxious", "nervous",
                "concern", "panic", "emergency", "urgent", "danger", "unsafe"],
    "satisfaction": ["thank", "thanks", "appreciat", "grateful", "excellent",
                     "great", "perfect", "happy", "satisf", "helpful"],
    "neutral": ["please", "kindly", "could you", "can you", "would like",
                "confirm", "request"],
}
NEGATIONS = {"not", "no", "never", "don't", "dont", "didn't", "didnt",
             "isn't", "isnt", "wasn't", "wasnt", "can't", "cant", "cannot",
             "won't", "wont", "hardly", "nahi"}

PATTERNS = {e: [re.compile(r"\b" + re.escape(t)) for t in terms]
            for e, terms in LEXICON.items()}
EMOTIONS = list(LEXICON.keys())


def lexicon_vector(text):
    """Counts of lexicon hits per emotion plus a few writing-style signals.

    A positive word preceded by a negation within three words ("not happy")
    is counted toward disappointment rather than satisfaction.
    """
    low = text.lower()
    tokens = re.findall(r"[a-z']+", low)
    counts = dict.fromkeys(EMOTIONS, 0.0)
    for emotion, pats in PATTERNS.items():
        for pat in pats:
            for m in pat.finditer(low):
                before = re.findall(r"[a-z']+", low[max(0, m.start() - 25):m.start()])[-3:]
                if emotion == "satisfaction" and NEGATIONS.intersection(before):
                    counts["disappointment"] += 1
                else:
                    counts[emotion] += 1

    n = max(len(tokens), 1)
    style = [
        min(text.count("!"), 5),                                  # exclamations
        min(len(re.findall(r"\b[A-Z]{3,}\b", text)), 5),          # shouting
        min(text.count("?"), 3),                                  # questions
        np.log1p(len(tokens)),                                    # length
        sum(1 for t in tokens if t in NEGATIONS) / n,             # negation rate
    ]
    return [counts[e] for e in EMOTIONS] + style


class LexiconFeatures(BaseEstimator, TransformerMixin):
    """Emotion lexicon counts and style signals as dense features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([lexicon_vector(t) for t in X], dtype=float)


class SourceModelFeatures(BaseEstimator, TransformerMixin):
    """Probabilities from a model trained on GoEmotions, used as features.

    The source model is already fitted on Reddit data and is never refitted
    here, so it cannot see complaint labels. Its six probabilities summarise
    what 28,000 examples taught it about emotion, without letting Reddit
    vocabulary dominate the complaint model's own TF-IDF features.
    """

    def __init__(self, source=None):
        self.source = source

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.source.predict_proba(list(X))
