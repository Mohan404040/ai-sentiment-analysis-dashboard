"""AI-Powered Sentiment Analysis Dashboard (Streamlit).

    streamlit run app.py
"""
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sentiment import LABELS, MODELS_DIR, ROOT
from sentiment.absa import analyse_aspects, analyse_aspects_batch, build_aspects
from sentiment.predict import SentimentPredictor, load_metrics, top_words

st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="💬", layout="wide")

COLORS = {"Negative": "#d62728", "Neutral": "#8c8c8c", "Positive": "#2ca02c"}
TEXT_COLUMN_GUESSES = ["text", "review", "reviews", "comment", "comments", "feedback",
                       "review_text", "content", "message", "tweet", "sentence"]
LABEL_COLUMN_GUESSES = ["label", "sentiment", "true_label", "airline_sentiment", "class"]


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner=False)
def get_predictor(name, stamp):
    return SentimentPredictor.load(name)  # `stamp` changes after retraining, so the model reloads


@st.cache_data(show_spinner=False)
def get_metrics(stamp):
    return load_metrics()


def metrics_stamp():
    path = MODELS_DIR / "metrics.json"
    return path.stat().st_mtime if path.exists() else 0


metrics = get_metrics(metrics_stamp())

if metrics is None:
    st.title("💬 Sentiment Analysis Dashboard")
    st.warning("No trained model found. Train the models first (about 1 minute). "
               "This downloads the Twitter US Airline Sentiment dataset if it is not already in data/.")
    if st.button("Train models now", type="primary"):
        from sentiment.model import run_training
        with st.spinner("Training Naive Bayes and Logistic Regression..."):
            run_training(verbose=False)
        get_metrics.clear()
        st.rerun()
    st.stop()

model_keys = list(metrics["models"])
display = {k: metrics["models"][k]["display_name"] for k in model_keys}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
    best = metrics["best_model"]
    choice = st.selectbox(
        "Classifier", model_keys, index=model_keys.index(best),
        format_func=lambda k: display[k] + ("  (best)" if k == best else ""),
    )
    m = metrics["models"][choice]["test"]
    c1, c2 = st.columns(2)
    c1.metric("Test accuracy", f"{m['accuracy']:.1%}")
    c2.metric("Macro F1", f"{m['f1_macro']:.3f}")
    st.caption(f"Trained on: {metrics['dataset']['source']}  \n"
               f"{metrics['dataset']['train_rows']:,} train / {metrics['dataset']['test_rows']:,} test texts  \n"
               f"Trained at {metrics['trained_at']}")
    st.divider()
    st.caption("Classes: " + ", ".join(m["labels"]))

predictor = get_predictor(choice, metrics_stamp())


def sentiment_badge(label, confidence=None):
    color = COLORS.get(label, "#1f77b4")
    conf = "" if confidence is None or pd.isna(confidence) else f" &nbsp;·&nbsp; {confidence:.0%} confident"
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;background:{color}22;"
        f"border-left:6px solid {color};font-size:1.4rem;font-weight:600;'>"
        f"{label}<span style='font-size:1rem;font-weight:400;'>{conf}</span></div>",
        unsafe_allow_html=True,
    )


def guess_column(columns, guesses):
    lower = {c.lower(): c for c in columns}
    for g in guesses:
        if g in lower:
            return lower[g]
    return None


st.title("💬 AI-Powered Sentiment Analysis Dashboard")
st.caption("Predicts whether feedback, reviews or comments are Positive, Negative or Neutral "
           "using TF-IDF features and a machine-learning classifier.")

tab_single, tab_csv, tab_absa, tab_perf, tab_about = st.tabs(
    ["📝 Single review", "📂 CSV upload", "🔍 Aspect analysis", "📊 Model performance", "ℹ️ About"]
)

# ---------------------------------------------------------------- single review
with tab_single:
    examples = {
        "(type your own)": "",
        "Positive example": "The crew were fantastic and the flight landed early. Great experience!",
        "Negative example": "Two hour delay and nobody at the gate could tell us anything. Not happy.",
        "Neutral example": "Is there a direct flight from Boston to Denver on Friday morning?",
        "Negation example": "The food was not good and the seat was not comfortable.",
    }
    pick = st.selectbox("Try an example", list(examples))
    text = st.text_area("Review / comment", value=examples[pick], height=120,
                        placeholder="Type or paste a review here...")
    if st.button("Analyse sentiment", type="primary") and text.strip():
        row = predictor.predict_one(text)
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Prediction")
            sentiment_badge(row["sentiment"], row["confidence"])
            if row["note"]:
                st.info("No usable words left after cleaning, so this is treated as Neutral.")
            st.markdown("**Text after preprocessing**")
            st.code(row["clean"] or "(empty)", language=None)
        with right:
            probs = pd.DataFrame({"Sentiment": predictor.classes,
                                  "Probability": [row[f"p_{c}"] for c in predictor.classes]})
            fig = px.bar(probs, x="Probability", y="Sentiment", orientation="h", color="Sentiment",
                         color_discrete_map=COLORS, range_x=[0, 1], text_auto=".0%")
            fig.update_layout(showlegend=False, height=230, margin=dict(l=0, r=0, t=30, b=0),
                              title="Class probabilities")
            st.plotly_chart(fig)
        expl = predictor.explain(text)
        if len(expl):
            st.markdown(f"**Words that influenced the prediction** (positive = pushed towards "
                        f"*{row['sentiment']}*, negative = pushed away)")
            expl["direction"] = expl["contribution"].apply(lambda v: "towards" if v > 0 else "away")
            fig = px.bar(expl.iloc[::-1], x="contribution", y="term", orientation="h", color="direction",
                         color_discrete_map={"towards": COLORS.get(row["sentiment"], "#1f77b4"),
                                             "away": "#bbbbbb"})
            fig.update_layout(height=60 + 32 * len(expl), margin=dict(l=0, r=0, t=10, b=0),
                              showlegend=False, yaxis_title=None)
            st.plotly_chart(fig)

# ---------------------------------------------------------------- CSV upload
with tab_csv:
    st.write("Upload a CSV with one review per row. A small sample file is included if you just want to try it.")
    up = st.file_uploader("CSV file", type=["csv"])
    data = None
    if up is not None:
        raw_bytes = up.getvalue()
        for enc in ("utf-8", "latin-1"):
            try:
                data = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
                break
            except UnicodeDecodeError:
                continue
    elif st.checkbox("Use the sample file (data/sample_reviews.csv)", value=True):
        data = pd.read_csv(ROOT / "data" / "sample_reviews.csv")

    if data is not None and len(data):
        cols = list(data.columns)
        c1, c2 = st.columns(2)
        guess = guess_column(cols, TEXT_COLUMN_GUESSES)
        text_col = c1.selectbox("Text column", cols, index=cols.index(guess) if guess else 0)
        label_guess = guess_column(cols, LABEL_COLUMN_GUESSES)
        label_opts = ["(none)"] + cols
        label_col = c2.selectbox("True label column (optional, to check accuracy)", label_opts,
                                 index=label_opts.index(label_guess) if label_guess else 0)
        max_rows = 20000
        if len(data) > max_rows:
            st.info(f"Using the first {max_rows:,} of {len(data):,} rows.")
            data = data.head(max_rows)

        data = data.dropna(subset=[text_col]).reset_index(drop=True)  # skip rows with an empty text cell
        if data.empty:
            st.warning("The selected text column has no text. Pick another column.")
        else:
            res = predictor.predict(data[text_col].tolist())
            out = data.copy()
            out["predicted_sentiment"] = res["sentiment"].values
            out["confidence"] = res["confidence"].round(3).values

            counts = res["sentiment"].value_counts().reindex(
                [l for l in LABELS if l in predictor.classes], fill_value=0)
            total = int(counts.sum())
            st.subheader("Overall sentiment")
            mcols = st.columns(len(counts) + 1)
            mcols[0].metric("Reviews", f"{total:,}")
            for col, (lab, n) in zip(mcols[1:], counts.items()):
                col.metric(lab, f"{n:,}", f"{n / total:.1%}", delta_color="off")

            g1, g2 = st.columns(2)
            pie = px.pie(names=counts.index, values=counts.values, color=counts.index,
                         color_discrete_map=COLORS, hole=0.45, title="Share of each sentiment")
            g1.plotly_chart(pie)
            bar = px.bar(x=counts.index, y=counts.values, color=counts.index, color_discrete_map=COLORS,
                         labels={"x": "Sentiment", "y": "Reviews"}, title="Reviews per sentiment", text_auto=True)
            bar.update_layout(showlegend=False)
            g2.plotly_chart(bar)

            st.subheader("Common words by sentiment")
            wcols = st.columns(len(counts))
            for col, lab in zip(wcols, counts.index):
                words = top_words(res.loc[res["sentiment"] == lab, "clean"], n=12)
                col.markdown(f"**{lab}**")
                if len(words):
                    fig = px.bar(words.iloc[::-1], x="count", y="word", orientation="h",
                                 color_discrete_sequence=[COLORS.get(lab)])
                    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
                    col.plotly_chart(fig)
                else:
                    col.caption("No reviews")

            st.subheader("Themes (aspects) mentioned")
            absa_df = analyse_aspects_batch(data[text_col].astype(str).tolist(), predictor)
            if len(absa_df):
                agg = absa_df.groupby(["aspect", "sentiment"]).size().reset_index(name="mentions")
                order = agg.groupby("aspect")["mentions"].sum().sort_values().index.tolist()
                fig = px.bar(agg, x="mentions", y="aspect", color="sentiment", orientation="h",
                             color_discrete_map=COLORS, category_orders={"aspect": order[::-1]})
                fig.update_layout(height=120 + 30 * len(order), yaxis_title=None, margin=dict(t=10))
                st.plotly_chart(fig)
            else:
                st.caption("No known aspects found. Add your own on the Aspect analysis tab.")

            if label_col != "(none)":
                from sentiment.data import normalise_label
                from sklearn.metrics import accuracy_score, f1_score
                truth = data[label_col].map(normalise_label)
                ok = truth.notna()
                st.subheader("Agreement with your labels")
                if not ok.any():
                    st.info("The chosen label column is empty.")
                else:
                    if not set(truth[ok]) <= set(LABELS):
                        st.warning("This column has values other than Negative / Neutral / Positive, "
                                   "so the accuracy below will be low. Pick the right label column.")
                    acc = accuracy_score(truth[ok], res.loc[ok, "sentiment"])
                    f1 = f1_score(truth[ok], res.loc[ok, "sentiment"], average="macro", zero_division=0)
                    a1, a2 = st.columns(2)
                    a1.metric("Accuracy", f"{acc:.1%}")
                    a2.metric("Macro F1", f"{f1:.3f}")
                    out["true_label"] = truth.values
                    out["correct"] = out["true_label"] == out["predicted_sentiment"]

            st.subheader("Predictions")
            filt = st.multiselect("Show sentiments", list(counts.index), default=list(counts.index))
            st.dataframe(out[out["predicted_sentiment"].isin(filt)], hide_index=True)
            st.download_button("Download results as CSV", out.to_csv(index=False).encode("utf-8"),
                               "sentiment_results.csv", "text/csv")

# ---------------------------------------------------------------- ABSA
with tab_absa:
    st.write("Aspect-Based Sentiment Analysis: instead of one sentiment for the whole review, "
             "find the sentiment about each aspect mentioned (camera, battery, staff, price...).")
    absa_text = st.text_area(
        "Review", height=110, key="absa_text",
        value="The camera is excellent, but the battery is poor. Delivery was quick, "
              "however the price is too expensive.")
    custom = st.text_input("Extra aspects to look for (comma separated, optional)",
                           placeholder="e.g. wifi, pool, keyboard")
    if st.button("Find aspects", type="primary") and absa_text.strip():
        aspects = build_aspects(custom)
        overall = predictor.predict_one(absa_text)
        st.markdown("**Overall sentiment**")
        sentiment_badge(overall["sentiment"], overall["confidence"])
        table = analyse_aspects(absa_text, predictor, aspects)
        st.markdown("**Sentiment per aspect**")
        if len(table):
            for _, r in table.iterrows():
                color = COLORS.get(r["sentiment"], "#1f77b4")
                st.markdown(
                    f"<div style='margin:6px 0;padding:8px 12px;border-radius:8px;"
                    f"border-left:5px solid {color};background:{color}18'>"
                    f"<b>{r['aspect']}</b>: <span style='color:{color};font-weight:600'>{r['sentiment']}</span>"
                    f" <span style='opacity:.7'>({r['confidence']:.0%})</span><br>"
                    f"<span style='opacity:.8'>“{r['clause']}”</span></div>",
                    unsafe_allow_html=True)
        else:
            st.info("No known aspects found. Add the aspects you care about in the box above.")
    with st.expander("How does this work?"):
        st.markdown(
            "1. The review is split into clauses at sentence ends and contrast words "
            "(*but, however, although...*).\n"
            "2. Each clause is searched for aspect keywords (a built-in list plus anything you add).\n"
            "3. The trained sentiment model predicts the sentiment of each clause that mentions an aspect.\n\n"
            "Limitations: aspects are found by keyword only, and a clause mentioning two aspects gives "
            "both the same sentiment.")

# ---------------------------------------------------------------- performance
with tab_perf:
    st.write("All numbers are on the held-out test set, which was not used for training or tuning.")
    rows = []
    for k in model_keys:
        t = metrics["models"][k]["test"]
        rows.append({"Model": display[k], "CV macro-F1": metrics["models"][k]["cv_f1_macro"],
                     "Accuracy": t["accuracy"], "Precision (macro)": t["precision_macro"],
                     "Recall (macro)": t["recall_macro"], "F1 (macro)": t["f1_macro"],
                     "F1 (weighted)": t["f1_weighted"]})
    comp = pd.DataFrame(rows).set_index("Model")
    st.subheader("Model comparison")
    st.dataframe(comp.style.format("{:.3f}").highlight_max(axis=0, color="#2ca02c33"))
    st.caption(f"Best model: **{display[best]}**, chosen by {metrics['selection']}.")
    long = comp.drop(columns=["CV macro-F1"]).reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", range_y=[0, 1], text_auto=".2f")
    st.plotly_chart(fig)

    st.subheader("Confusion matrices")
    ccols = st.columns(len(model_keys))
    for col, k in zip(ccols, model_keys):
        t = metrics["models"][k]["test"]
        cm = pd.DataFrame(t["confusion_matrix"], index=t["labels"], columns=t["labels"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="Count"), title=display[k])
        fig.update_layout(coloraxis_showscale=False)
        col.plotly_chart(fig)

    st.subheader("Per-class results")
    k = st.radio("Model", model_keys, format_func=lambda k: display[k], horizontal=True, index=model_keys.index(best))
    t = metrics["models"][k]["test"]
    per = pd.DataFrame({lab: t["per_class"][lab] for lab in t["labels"]}).T
    per = per.rename(columns={"f1-score": "f1", "support": "test texts"})
    st.dataframe(per.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}", "test texts": "{:.0f}"}))

    st.subheader("Dataset")
    summ = metrics["dataset"]["raw_summary"]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Rows", f"{summ['rows']:,}")
    d2.metric("Classes", summ["n_classes"])
    d3.metric("Duplicate texts removed", f"{summ['duplicate_texts']:,}")
    d4.metric("Avg words / text", summ["avg_words"])
    cc = pd.Series(summ["class_counts"])
    fig = px.bar(x=cc.index, y=cc.values, color=cc.index, color_discrete_map=COLORS, text_auto=True,
                 labels={"x": "Class", "y": "Texts"}, title="Class balance (imbalanced, so macro-F1 matters)")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

    pred_path = MODELS_DIR / "test_predictions.csv"
    if pred_path.exists():
        st.subheader("Incorrect predictions (error analysis)")
        tp = pd.read_csv(pred_path)
        wrong = tp[tp["label"] != tp[f"pred_{k}"]][["text", "label", f"pred_{k}"]]
        wrong = wrong.rename(columns={"label": "actual", f"pred_{k}": "predicted"})
        st.caption(f"{len(wrong):,} of {len(tp):,} test texts are misclassified by {display[k]}.")
        pair = wrong.groupby(["actual", "predicted"]).size().reset_index(name="count").sort_values("count", ascending=False)
        st.dataframe(pair, hide_index=True)
        st.dataframe(wrong.sample(min(25, len(wrong)), random_state=0), hide_index=True)

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown(f"""
**Pipeline**

`User input (text / CSV)` → `Preprocessing` → `TF-IDF features` → `Classifier` → `Dashboard` → `Evaluation`

- **Dataset:** {metrics['dataset']['source']} - human-labelled Negative / Neutral / Positive customer feedback
  (a genuine 3-class dataset; no Neutral labels were invented).
- **Preprocessing:** lower-case, remove URLs and @mentions, expand contractions (*isn't → is not*),
  keep letters only, remove stopwords **but keep negations** (*not, no, never...*), lemmatise (NLTK WordNet).
- **Features:** TF-IDF on unigrams and bigrams (bigrams capture *not good*).
- **Models:** Multinomial Naive Bayes and Logistic Regression, tuned with 5-fold cross-validation on macro-F1.
- **Evaluation:** accuracy, precision, recall, F1 (macro and weighted) and confusion matrix on a held-out 20% test set.
- **Enhancement:** keyword + clause based Aspect-Based Sentiment Analysis using the trained model.

**Known limitations**
- The model learned from airline tweets, so it is strongest on service / travel feedback. Product reviews
  work for clear opinions but neutral factual product sentences can be misread. Retrain on your own labelled
  data with `python train.py --csv your.csv --text-col review --label-col sentiment`.
- Sarcasm and mixed opinions in one sentence are hard for bag-of-words models.
""")
