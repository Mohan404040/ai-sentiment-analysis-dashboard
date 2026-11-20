"""Aspect-Based Sentiment Analysis (ABSA) - the enhancement from section 6 of the guide.

Simple, explainable approach that reuses the trained 3-class model:
  1. Split the review into clauses at sentence ends and contrast words
     ("but", "however", "although", ...).
  2. Find aspect keywords in each clause (a dictionary of common aspects, plus any
     aspects you type in).
  3. Predict the sentiment of each clause that mentions an aspect with the trained model.

"The camera is excellent, but the battery is poor."
    -> Camera: Positive   ("The camera is excellent")
    -> Battery: Negative  ("the battery is poor")

Limitations: aspects are found by keywords (no synonym detection), and a clause that
mentions two aspects gives both the same sentiment.
"""
import re

import pandas as pd

DEFAULT_ASPECTS = {
    "Service / Staff": ["service", "staff", "crew", "attendant", "attendants", "agent", "agents",
                        "employee", "support", "customer service", "waiter", "rep", "helpful", "rude"],
    "Delay / Timing": ["delay", "delayed", "delays", "late", "on time", "wait", "waiting",
                       "hour", "hours", "cancel", "cancelled", "canceled", "cancellation"],
    "Price / Value": ["price", "prices", "cost", "fee", "fees", "expensive", "cheap",
                      "refund", "money", "value", "worth"],
    "Baggage": ["bag", "bags", "baggage", "luggage", "suitcase"],
    "Seat / Comfort": ["seat", "seats", "legroom", "comfort", "comfortable", "space"],
    "Food / Drink": ["food", "meal", "meals", "snack", "snacks", "drink", "drinks", "taste", "coffee"],
    "Website / App": ["website", "site", "app", "online", "booking", "check-in", "checkin"],
    "Delivery": ["delivery", "shipping", "shipped", "arrived", "package", "courier"],
    "Quality": ["quality", "build", "material", "durable", "broke", "broken"],
    "Camera": ["camera", "photo", "photos", "picture", "pictures", "lens"],
    "Battery": ["battery", "charging", "charger"],
    "Screen / Display": ["screen", "display", "brightness"],
    "Sound": ["sound", "audio", "speaker", "speakers", "volume"],
    "Performance": ["performance", "speed", "lag", "laggy", "slow", "fast"],
}

CLAUSE_SPLIT_RE = re.compile(
    r"[.!?;\n]+|,?\s*\b(?:but|however|although|though|whereas|except)\b",
    flags=re.IGNORECASE,
)


def split_clauses(text):
    parts = CLAUSE_SPLIT_RE.split(str(text))
    return [p.strip(" ,-") for p in parts if p and len(p.strip(" ,-").split()) > 0]


def build_aspects(custom=None, use_defaults=True):
    """Merge default aspects with custom ones. custom can be 'wifi, pool' or {'Wifi': ['wifi']}."""
    aspects = dict(DEFAULT_ASPECTS) if use_defaults else {}
    if isinstance(custom, str):
        custom = [c.strip() for c in custom.split(",") if c.strip()]
    if isinstance(custom, (list, tuple)):
        custom = {c.title(): [c.lower()] for c in custom}
    for name, words in (custom or {}).items():
        aspects[name] = [w.lower() for w in words]
    return aspects


def _find_aspects(clause, aspects):
    low = clause.lower()
    hits = []
    for aspect, words in aspects.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", low):
                hits.append((aspect, w))
                break
    return hits


def analyse_aspects(text, predictor, aspects=None):
    """One row per (aspect, clause): aspect, keyword, clause, sentiment, confidence."""
    aspects = aspects or DEFAULT_ASPECTS
    rows = []
    for clause in split_clauses(text):
        for aspect, word in _find_aspects(clause, aspects):
            rows.append({"aspect": aspect, "keyword": word, "clause": clause})
    if not rows:
        return pd.DataFrame(columns=["aspect", "keyword", "clause", "sentiment", "confidence"])
    df = pd.DataFrame(rows)
    pred = predictor.predict(df["clause"].tolist())
    df["sentiment"] = pred["sentiment"].values
    df["confidence"] = pred["confidence"].values
    return df


def analyse_aspects_batch(texts, predictor, aspects=None):
    """ABSA over many reviews (one clause-level prediction call for speed)."""
    aspects = aspects or DEFAULT_ASPECTS
    rows = []
    for i, text in enumerate(texts):
        for clause in split_clauses(text):
            for aspect, word in _find_aspects(clause, aspects):
                rows.append({"review_id": i, "aspect": aspect, "keyword": word, "clause": clause})
    if not rows:
        return pd.DataFrame(columns=["review_id", "aspect", "keyword", "clause", "sentiment"])
    df = pd.DataFrame(rows)
    df["sentiment"] = predictor.predict(df["clause"].tolist())["sentiment"].values
    return df
