"""Text preprocessing.

Steps: lower-case, decode HTML entities, turn emoticons into words, remove URLs and
@mentions, expand contractions ("isn't" -> "is not"), keep letters only, shorten
stretched words ("soooo" -> "soo"), remove stopwords, lemmatise.

Negation words ("not", "no", "never", ...) are kept on purpose: "good" and "not good"
mean opposite things, and the TF-IDF bigrams can only learn that if "not" survives.
"""
import html
import re
from functools import lru_cache

NEGATIONS = {
    "no", "not", "nor", "never", "none", "nobody", "nothing", "neither",
    "nowhere", "cannot", "without", "hardly", "barely",
}

# Emoticons carry sentiment but would be deleted with the punctuation, so keep them as words.
EMOTICONS = {
    ":)": " smileface ", ":-)": " smileface ", ":d": " smileface ", ":-d": " smileface ",
    ";)": " smileface ", "<3": " heartface ", ":(": " sadface ", ":-(": " sadface ",
    ":/": " sadface ", ":'(": " sadface ",
}

CONTRACTIONS = [
    (r"\bwon't\b", "will not"), (r"\bcan't\b", "can not"), (r"\bain't\b", "is not"),
    (r"\bshan't\b", "shall not"), (r"n't\b", " not"), (r"'re\b", " are"),
    (r"'m\b", " am"), (r"'ll\b", " will"), (r"'ve\b", " have"), (r"'d\b", " would"),
    (r"'s\b", ""),
]

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_LETTER_RE = re.compile(r"[^a-z\s]")
STRETCH_RE = re.compile(r"(\w)\1{2,}")

_STOPWORDS = None
_LEMMATIZER = None


def _nltk_resource(loader, packages):
    """Run loader(); if NLTK data is missing, download it once and retry."""
    import nltk
    try:
        return loader()
    except LookupError:
        for pkg in packages:
            nltk.download(pkg, quiet=True)
        return loader()


def get_stopwords():
    """English stopwords from NLTK (scikit-learn's list if NLTK data is unavailable),
    minus the negation words."""
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            from nltk.corpus import stopwords
            words = set(_nltk_resource(lambda: stopwords.words("english"), ["stopwords"]))
        except Exception:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            words = set(ENGLISH_STOP_WORDS)
        _STOPWORDS = words - NEGATIONS
    return _STOPWORDS


def get_lemmatizer():
    """WordNet lemmatiser, or a no-op if WordNet cannot be loaded."""
    global _LEMMATIZER
    if _LEMMATIZER is None:
        try:
            from nltk.stem import WordNetLemmatizer
            lem = WordNetLemmatizer()
            _nltk_resource(lambda: lem.lemmatize("flights"), ["wordnet", "omw-1.4"])
            _LEMMATIZER = lem.lemmatize
        except Exception:
            _LEMMATIZER = lambda word: word
    return _LEMMATIZER


@lru_cache(maxsize=50000)
def _lemma(word):
    return get_lemmatizer()(word)


def clean_text(text, remove_stopwords=True, lemmatize=True):
    """Clean one text and return the processed string."""
    if text is None or (isinstance(text, float) and text != text):  # None or NaN
        return ""
    text = html.unescape(str(text)).lower().replace("’", "'").replace("‘", "'")
    text = URL_RE.sub(" ", text)  # before emoticons, so "https://" is not read as ":/"
    text = MENTION_RE.sub(" ", text)
    for emo, word in EMOTICONS.items():
        text = text.replace(emo, word)
    for pattern, repl in CONTRACTIONS:
        text = re.sub(pattern, repl, text)
    text = NON_LETTER_RE.sub(" ", text)
    text = STRETCH_RE.sub(r"\1\1", text)

    stop = get_stopwords() if remove_stopwords else set()
    tokens = []
    for tok in text.split():
        if tok in stop or (len(tok) < 2 and tok not in NEGATIONS):
            continue
        tokens.append(_lemma(tok) if lemmatize else tok)
    return " ".join(tokens)


def clean_texts(texts, **kwargs):
    """Clean a list / Series of texts."""
    return [clean_text(t, **kwargs) for t in texts]
