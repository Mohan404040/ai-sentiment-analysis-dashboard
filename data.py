"""Load the labelled dataset and check it (classes, missing values, duplicates).

Default dataset: Twitter US Airline Sentiment (14,640 customer tweets, labelled by
human annotators as negative / neutral / positive). It is a genuine 3-class dataset,
so no Neutral labels are invented.

Where it is loaded from, in order:
  1. data/Tweets.csv or data/airline.parquet inside the project
  2. any Tweets.csv under /kaggle/input (when the Kaggle dataset
     "crowdflower/twitter-airline-sentiment" is attached to the notebook)
  3. a download of the same data from the Hugging Face Hub (cached to data/airline.parquet)

You can also train on your own CSV with load_dataset(csv_path=..., text_col=..., label_col=...).
"""
import glob
import urllib.request
from pathlib import Path

import pandas as pd

from . import DATA_DIR, LABELS

AIRLINE_URLS = [
    "https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "https://huggingface.co/api/datasets/osanseviero/twitter-airline-sentiment/"
    "parquet/default/train/0.parquet",
]

LABEL_ALIASES = {
    "negative": "Negative", "neg": "Negative",
    "neutral": "Neutral", "neu": "Neutral",
    "positive": "Positive", "pos": "Positive",
}


def normalise_label(value):
    """Map label spellings such as 'negative' or 'NEG' to 'Negative'.
    Missing values (NaN / None / empty) stay missing so they can be dropped."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    return LABEL_ALIASES.get(text.lower(), text.title())


def _local_airline_files():
    files = [DATA_DIR / "Tweets.csv", DATA_DIR / "airline.parquet",
             Path("/content/Tweets.csv"), Path("Tweets.csv")]  # /content = Colab upload folder
    files += [Path(p) for p in glob.glob("/kaggle/input/**/Tweets.csv", recursive=True)]
    return [f for f in files if f.exists()]


def _download_airline():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / "airline.parquet"
    errors = []
    for url in AIRLINE_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sentiment-dashboard"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                content = resp.read()
            tmp = target.with_suffix(".part")
            tmp.write_bytes(content)
            pd.read_parquet(tmp, columns=["text", "airline_sentiment"])  # reject HTML / broken files
            tmp.replace(target)  # only a valid file is kept (and cached for the next run)
            return target
        except Exception as exc:  # try the next mirror
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        "Could not download the dataset. Turn on internet access (Kaggle: Settings -> "
        "Internet), or attach the Kaggle dataset 'crowdflower/twitter-airline-sentiment', "
        "or download Tweets.csv from that Kaggle page and put it in the data/ folder "
        "(on Colab: upload it in the Files panel, it is found automatically).\n" + "\n".join(errors)
    )


def load_airline():
    """Return the airline dataset as a DataFrame with columns text, label."""
    files = _local_airline_files()
    path = files[0] if files else _download_airline()
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if not {"text", "airline_sentiment"} <= set(raw.columns):
        raise ValueError(f"{path} needs columns 'text' and 'airline_sentiment'; found {list(raw.columns)}")
    df = raw[["text", "airline_sentiment"]].rename(columns={"airline_sentiment": "label"})
    df["label"] = df["label"].map(normalise_label)
    df.attrs["source"] = f"Twitter US Airline Sentiment ({path.name})"
    return df


def load_dataset(csv_path=None, text_col="text", label_col="label"):
    """Load the default dataset, or your own CSV with a text column and a label column."""
    if csv_path is None:
        return load_airline()
    raw = pd.read_csv(csv_path)
    missing = [c for c in (text_col, label_col) if c not in raw.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found. CSV columns: {list(raw.columns)}")
    df = raw[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["label"] = df["label"].map(normalise_label)
    unexpected = sorted(set(df["label"].dropna()) - set(LABELS))
    if unexpected:
        print(f"Warning: labels other than Negative/Neutral/Positive found: {unexpected}. "
              "They are kept as separate classes - map them to the three labels if that is not intended.")
    df.attrs["source"] = Path(csv_path).name
    return df


def check_dataset(df):
    """Summary used in step 2: size, class balance, missing values and duplicates."""
    counts = df["label"].value_counts()
    order = [l for l in LABELS if l in counts.index] + [l for l in counts.index if l not in LABELS]
    counts = counts.reindex(order)
    words = df["text"].fillna("").astype(str).str.split().str.len()
    return {
        "rows": int(len(df)),
        "n_classes": int(counts.size),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "class_percent": {k: round(100 * v / counts.sum(), 1) for k, v in counts.items()},
        "missing_text": int(df["text"].isna().sum()),
        "missing_label": int(df["label"].isna().sum()),
        "duplicate_texts": int(df["text"].duplicated().sum()),
        "avg_words": round(float(words.mean()), 1),
    }


def prepare_dataset(df):
    """Drop rows with missing text/label, empty text and duplicate texts."""
    out = df.dropna(subset=["text", "label"]).copy()
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"] != ""]
    out = out.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    out.attrs.update(df.attrs)
    return out
