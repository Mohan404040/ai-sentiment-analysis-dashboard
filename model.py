"""Training, comparison and evaluation of the two classifiers.

Both models are scikit-learn Pipelines: TF-IDF features -> classifier.
  * Multinomial Naive Bayes
  * Logistic Regression
Hyper-parameters are tuned with 5-fold cross-validation on the training set using
macro F1 (fair to the smaller classes). The best model is chosen by that
cross-validation score, so the held-out test set is only used for the final report.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from . import LABELS, MODELS_DIR
from .preprocess import clean_texts

MODEL_NAMES = {
    "naive_bayes": "Multinomial Naive Bayes",
    "logistic_regression": "Logistic Regression",
}


def make_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)


def build_search(name, cv_folds=5, n_jobs=-1, seed=42):
    """GridSearchCV object for one model."""
    if name == "naive_bayes":
        pipe = Pipeline([("tfidf", make_tfidf()), ("clf", MultinomialNB())])
        grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "clf__alpha": [0.1, 0.3, 0.5, 1.0],
            "clf__fit_prior": [True, False],
        }
    elif name == "logistic_regression":
        pipe = Pipeline([("tfidf", make_tfidf()), ("clf", LogisticRegression(max_iter=3000))])
        grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "clf__C": [0.5, 1, 2, 5, 10],
            "clf__class_weight": [None, "balanced"],
        }
    else:
        raise ValueError(f"Unknown model {name!r}; use one of {list(MODEL_NAMES)}")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    return GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=n_jobs)


def add_clean_column(df):
    out = df.copy()
    out["clean"] = clean_texts(out["text"])
    return out


def split_data(df, test_size=0.2, seed=42):
    """Stratified train / test split (keeps the class balance the same in both)."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def ordered_labels(y):
    present = set(y)
    return [l for l in LABELS if l in present] + sorted(present - set(LABELS))


def evaluate(model, texts_clean, y_true):
    """Accuracy, precision, recall, F1 (macro and weighted), per-class report and
    confusion matrix on a held-out set."""
    y_pred = model.predict(texts_clean)
    labels = ordered_labels(list(y_true) + list(y_pred))
    out = {"labels": labels, "accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = float(p), float(r), float(f)
    out["per_class"] = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return out, y_pred


def train_models(train_df, names=tuple(MODEL_NAMES), verbose=True):
    """Tune and fit each model. Returns {name: {"model", "best_params", "cv_f1_macro", "fit_seconds"}}."""
    results = {}
    for name in names:
        start = time.time()
        search = build_search(name)
        search.fit(train_df["clean"], train_df["label"])
        results[name] = {
            "model": search.best_estimator_,
            "best_params": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in search.best_params_.items()},
            "cv_f1_macro": float(search.best_score_),
            "fit_seconds": round(time.time() - start, 1),
        }
        if verbose:
            print(f"{MODEL_NAMES[name]:<26} CV macro-F1 = {search.best_score_:.4f}  "
                  f"best params = {search.best_params_}  ({results[name]['fit_seconds']}s)")
    return results


def evaluate_models(results, test_df):
    """Add test metrics and test predictions to each result."""
    for res in results.values():
        res["test"], res["test_pred"] = evaluate(res["model"], test_df["clean"], test_df["label"])
    return results


def choose_best(results):
    return max(results, key=lambda n: results[n]["cv_f1_macro"])


def comparison_table(results):
    rows = []
    for name, res in results.items():
        t = res["test"]
        rows.append({
            "Model": MODEL_NAMES.get(name, name),
            "CV macro-F1": res["cv_f1_macro"],
            "Accuracy": t["accuracy"],
            "Precision (macro)": t["precision_macro"],
            "Recall (macro)": t["recall_macro"],
            "F1 (macro)": t["f1_macro"],
            "F1 (weighted)": t["f1_weighted"],
        })
    return pd.DataFrame(rows).set_index("Model").round(4)


def save_artifacts(results, best_name, test_df, dataset_info, out_dir=MODELS_DIR):
    """Save models (joblib), metrics.json and test_predictions.csv for the dashboard."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "best_model": best_name,
        "selection": "highest 5-fold cross-validation macro-F1 on the training set",
        "dataset": dataset_info,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_version": sklearn.__version__,
        "models": {},
    }
    preds = test_df[["text", "clean", "label"]].copy()
    for name, res in results.items():
        joblib.dump(res["model"], out_dir / f"{name}.joblib")
        metrics["models"][name] = {
            "display_name": MODEL_NAMES.get(name, name),
            "best_params": res["best_params"],
            "cv_f1_macro": res["cv_f1_macro"],
            "fit_seconds": res["fit_seconds"],
            "test": res["test"],
        }
        preds[f"pred_{name}"] = res["test_pred"]
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    preds.to_csv(out_dir / "test_predictions.csv", index=False)
    return out_dir


def run_training(csv_path=None, text_col="text", label_col="label", test_size=0.2,
                 out_dir=MODELS_DIR, verbose=True):
    """Full pipeline: load -> check -> clean -> split -> train -> evaluate -> save."""
    from .data import check_dataset, load_dataset, prepare_dataset

    say = print if verbose else (lambda *a, **k: None)
    raw = load_dataset(csv_path, text_col, label_col)
    info = check_dataset(raw)
    say(f"Loaded {info['rows']} rows from {raw.attrs.get('source')}: {info['class_counts']}")
    df = add_clean_column(prepare_dataset(raw))
    df = df[df["clean"] != ""].reset_index(drop=True)
    say(f"After removing missing / duplicate / empty texts: {len(df)} rows")
    if df["label"].nunique() < 2:
        raise ValueError("Need at least two classes to train.")

    train_df, test_df = split_data(df, test_size=test_size)
    say(f"Train: {len(train_df)}  Test: {len(test_df)}")
    results = evaluate_models(train_models(train_df, verbose=verbose), test_df)
    best = choose_best(results)
    dataset_info = {
        "source": raw.attrs.get("source"),
        "raw_summary": info,
        "rows_used": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_size": test_size,
        "train_class_counts": {k: int(v) for k, v in train_df["label"].value_counts().items()},
    }
    save_artifacts(results, best, test_df, dataset_info, out_dir)
    say("\n" + comparison_table(results).to_string())
    say(f"\nBest model (by CV macro-F1): {MODEL_NAMES[best]}. Saved to {Path(out_dir).resolve()}")
    return results, best, train_df, test_df
