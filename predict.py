"""Load a trained model and predict sentiment for new texts."""
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import MODELS_DIR
from .preprocess import NEGATIONS, clean_texts


def load_metrics(models_dir=MODELS_DIR):
    path = Path(models_dir) / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def models_available(models_dir=MODELS_DIR):
    return load_metrics(models_dir) is not None


class SentimentPredictor:
    def __init__(self, model, name=""):
        self.model = model
        self.name = name
        self.classes = list(model.classes_)

    @classmethod
    def load(cls, name=None, models_dir=MODELS_DIR):
        """Load a saved model; name=None loads the best one."""
        metrics = load_metrics(models_dir)
        if metrics is None:
            raise FileNotFoundError(f"No trained models in {models_dir}. Run: python train.py")
        name = name or metrics["best_model"]
        return cls(joblib.load(Path(models_dir) / f"{name}.joblib"), name)

    def predict(self, texts):
        """DataFrame with text, clean, sentiment, confidence and one probability column per class.

        A text with no usable words after cleaning (only emojis, numbers or stopwords) carries
        no sentiment the model can read, so it is labelled Neutral and marked in `note`."""
        texts = ["" if (t is None or (isinstance(t, float) and t != t)) else str(t) for t in texts]  # NaN -> ""
        cleaned = clean_texts(texts)
        proba = self.model.predict_proba(cleaned)
        out = pd.DataFrame({"text": texts, "clean": cleaned})
        out["sentiment"] = [self.classes[i] for i in proba.argmax(axis=1)]
        out["confidence"] = proba.max(axis=1)
        for i, cls in enumerate(self.classes):
            out[f"p_{cls}"] = proba[:, i]
        empty = out["clean"] == ""
        out["note"] = np.where(empty, "no usable words", "")
        if empty.any() and "Neutral" in self.classes:
            out.loc[empty, "sentiment"] = "Neutral"
            out.loc[empty, "confidence"] = np.nan
        return out

    def predict_one(self, text):
        return self.predict([text]).iloc[0]

    def explain(self, text, top_n=8):
        """Words that pushed the prediction towards the predicted class (and away from it).

        Contribution = TF-IDF value x class weight. Weights are the Logistic Regression
        coefficients (minus their mean over classes), or for Naive Bayes the class log-probability minus its mean over classes."""
        tfidf = self.model.named_steps["tfidf"]
        clf = self.model.named_steps["clf"]
        clean = clean_texts([text])[0]
        x = tfidf.transform([clean])
        if x.nnz == 0:
            return pd.DataFrame(columns=["term", "contribution"])
        cls_idx = int(self.model.predict_proba([clean]).argmax())
        if hasattr(clf, "coef_"):
            coef = clf.coef_
            # multi-class: only differences between classes matter (softmax ignores a shared shift),
            # so compare the predicted class with the average of all classes
            weights = (coef[0] if cls_idx == 1 else -coef[0]) if coef.shape[0] == 1 else coef[cls_idx] - coef.mean(axis=0)
        else:
            logp = clf.feature_log_prob_
            weights = logp[cls_idx] - logp.mean(axis=0)
        idx = x.indices
        contrib = x.data * weights[idx]
        terms = tfidf.get_feature_names_out()[idx]
        df = pd.DataFrame({"term": terms, "contribution": contrib})
        df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)
        return df.head(top_n).reset_index(drop=True)


def top_words(cleaned_texts, n=15, exclude=NEGATIONS):
    """Most frequent words in already-cleaned texts (used for 'common themes')."""
    counter = Counter(w for t in cleaned_texts for w in str(t).split()
                      if w not in exclude and len(w) > 2)
    return pd.DataFrame(counter.most_common(n), columns=["word", "count"])
