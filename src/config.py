"""
Central configuration for the Complaint Emotion & Urgency Analyzer.

Every path and constant lives here so no other module hard-codes a filename.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

for d in (PROCESSED, REPORTS, MODELS):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Target classes
# ---------------------------------------------------------------------------
# The six emotions a complaint-handling team actually needs to tell apart.
# Chosen before looking at the data, from the triage use case.
EMOTIONS = [
    "anger",
    "frustration",
    "disappointment",
    "anxiety",
    "satisfaction",
    "neutral",
]

# ---------------------------------------------------------------------------
# GoEmotions -> project label mapping
# ---------------------------------------------------------------------------
# GoEmotions (Demszky et al., ACL 2020) annotates Reddit comments with 27 fine
# grained emotions plus neutral. Several of those collapse naturally onto our
# six triage classes. Emotions with no clear complaint-handling meaning
# (amusement, curiosity, love, pride, surprise, ...) are dropped rather than
# forced into a bucket, because a wrong mapping is worse than less data.
GOEMOTIONS_MAP = {
    # anger: hostile, hot emotion directed at the company
    "anger": "anger",
    "disgust": "anger",
    # frustration: repeated friction, irritation, disapproval of how it was handled
    "annoyance": "frustration",
    "disapproval": "frustration",
    # disappointment: unmet expectation, deflation rather than heat
    "disappointment": "disappointment",
    "sadness": "disappointment",
    "grief": "disappointment",
    # anxiety: the customer is worried about a consequence
    "fear": "anxiety",
    "nervousness": "anxiety",
    # satisfaction: positive feedback, thanks, praise
    "gratitude": "satisfaction",
    "admiration": "satisfaction",
    "approval": "satisfaction",
    "joy": "satisfaction",
    "relief": "satisfaction",
    # neutral: informational request, no emotional content
    "neutral": "neutral",
}

# Neutral is about a third of GoEmotions. Left alone it dominates training and
# the classifier learns to answer "neutral" for everything. Capped here as a
# fraction of the largest non-neutral class.
NEUTRAL_CAP_RATIO = 1.0

MIN_TOKENS = 3      # drop fragments too short to carry an emotion
MAX_CHARS = 1000    # drop pathological outliers

TEST_SIZE = 0.15
VAL_SIZE = 0.15
