# AI-Powered Sentiment Analysis Dashboard

A web app that reads feedback, reviews or comments and predicts whether the sentiment is
**Positive, Negative or Neutral**. It compares **Multinomial Naive Bayes** and
**Logistic Regression** on **TF-IDF** features and shows everything in a **Streamlit** dashboard
with charts, model evaluation and aspect-based sentiment.

```
User input (text / CSV) → Preprocessing → TF-IDF + classifier → Dashboard → Evaluation
```

## Run it on Google Colab or Kaggle (easiest)

Everything is in **`Sentiment_Analysis.ipynb`**. It is self-contained: it writes the project files,
trains the models and starts the dashboard.

**Colab**
1. Go to colab.research.google.com → *File → Upload notebook* → choose `Sentiment_Analysis.ipynb`.
2. *Runtime → Run all*. A normal CPU runtime is enough (training takes about a minute).
3. The last part prints a link like `https://xxxx.trycloudflare.com` - open it to use the dashboard.
   (Colab also shows a second link through its own proxy.)

**Kaggle**
1. kaggle.com → *Create → New Notebook* → *File → Import Notebook* → upload `Sentiment_Analysis.ipynb`.
2. In the right panel, **Settings → Internet → On** (needed for installing Streamlit, downloading the data
   and the dashboard link). Optional: *Add Input* → dataset `crowdflower/twitter-airline-sentiment`
   (then the data is read from there instead of downloaded).
3. *Run All*, then open the printed `trycloudflare.com` link.

If the dataset download fails (it comes from the Hugging Face Hub), download `Tweets.csv` from
kaggle.com/datasets/crowdflower/twitter-airline-sentiment and put it in the Colab *Files* panel
(`/content/Tweets.csv`) or in the project's `data/` folder - it is picked up automatically.

If the link does not load immediately, wait 10-20 seconds and refresh. The dashboard keeps running while
the notebook session is active. Section 10 of the notebook downloads a zip of the project **with the trained models**.

## Run it on your own computer

```bash
pip install -r requirements.txt
python train.py          # downloads the dataset, trains and compares both models (~1 min)
streamlit run app.py     # opens the dashboard at http://localhost:8501
```

The dashboard also has a *Train models now* button if no trained model is found.

## Dataset

**Twitter US Airline Sentiment** - 14,640 customer tweets about six US airlines, labelled by human
annotators as negative (63%), neutral (21%) or positive (16%). It is a genuine 3-class dataset, so no
Neutral labels are created artificially. Because the classes are imbalanced, the project reports macro
precision, recall and F1 as well as accuracy.

Use your own labelled data instead:

```bash
python train.py --csv my_reviews.csv --text-col review --label-col sentiment
```

or set `CUSTOM_CSV` in step 1 of the notebook. Labels such as `positive`, `NEG`, `Neutral` are normalised automatically.

## Method

| Step | What happens |
|---|---|
| Data checks | class balance, missing values, duplicate texts (213 duplicates removed) |
| Preprocessing | lower-case, remove URLs and @mentions, emoticons → words, expand contractions (*isn't → is not*), letters only, shorten *sooooo → soo*, remove stopwords **but keep negations** (*not, no, never, …*), WordNet lemmatisation |
| Features | TF-IDF, unigrams + bigrams (bigrams let the model learn *not good*), sublinear TF |
| Models | Multinomial Naive Bayes and Logistic Regression in scikit-learn pipelines |
| Tuning | 5-fold stratified cross-validation on the training set, scored by macro-F1 |
| Model choice | best cross-validation macro-F1 (the test set is never used for choosing) |
| Evaluation | 80/20 stratified split; accuracy, precision, recall, F1 (macro and weighted), confusion matrix, error analysis |

## Results (held-out test set, 2,882 tweets)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|---|---|---|---|---|---|
| Multinomial Naive Bayes | 0.764 | 0.695 | 0.698 | 0.696 | 0.763 |
| **Logistic Regression** | **0.773** | **0.711** | **0.740** | **0.724** | **0.778** |

Logistic Regression (C = 1, balanced class weights, unigrams + bigrams) is the best model.
Per class (Logistic Regression): Negative F1 0.85, Positive F1 0.72, Neutral F1 0.60.
Neutral is the hardest class: questions and factual tweets often contain complaint words.
Numbers can differ very slightly with other library versions.

## Dashboard

| Tab | What it shows |
|---|---|
| Single review | predicted label, confidence, class probabilities, the words that drove the prediction, the cleaned text |
| CSV upload | prediction per row, counts and percentages, pie and bar charts, common words per sentiment, aspect themes, accuracy against your labels if the CSV has a label column, download of the results. A sample file is included. |
| Aspect analysis | sentiment per aspect (camera, battery, staff, price, delivery, …) plus your own aspects |
| Model performance | comparison table and chart, confusion matrices, per-class precision / recall / F1, class balance, misclassified test examples |

Use the sidebar to switch between Naive Bayes and Logistic Regression.

## Enhancement: Aspect-Based Sentiment Analysis

*"The camera is excellent, but the battery is poor."* → Camera: **Positive**, Battery: **Negative**

The review is split into clauses at sentence ends and contrast words (*but, however, although, …*),
aspect keywords are looked up in each clause, and the trained model predicts the sentiment of each clause.
This is a simple, explainable baseline: aspects are matched by keyword only, and two aspects in the same
clause get the same sentiment.

## Project structure

```
sentiment-dashboard/
├── Sentiment_Analysis.ipynb   # Colab / Kaggle notebook: all steps + launches the dashboard
├── app.py                     # Streamlit dashboard
├── train.py                   # command-line training
├── requirements.txt
├── data/
│   └── sample_reviews.csv     # 25 hand-written example reviews for the CSV tab
├── sentiment/
│   ├── data.py                # load + check the dataset
│   ├── preprocess.py          # text cleaning (keeps negations)
│   ├── model.py               # TF-IDF pipelines, tuning, evaluation, saving
│   ├── predict.py             # prediction, explanation, common words
│   └── absa.py                # aspect-based sentiment
└── models/                    # created by training: *.joblib, metrics.json, test_predictions.csv
```

## Limitations

- The model learned from airline tweets, so it is strongest on service and travel feedback. Clear product
  opinions work (*"I love this phone"* → Positive), but neutral factual product sentences can be misread
  (*"The package arrived on Tuesday"* leans Negative because in airline tweets "arrived" often appears in complaints).
  Retrain on a labelled dataset from your own domain for best results.
- Bag-of-words models do not understand sarcasm (*"great, another delay"*) or long mixed opinions.
- Next steps could be a transformer model (e.g. a fine-tuned BERT / RoBERTa) and a learned ABSA model; compare
  them with this baseline. When writing the report, describe the research gap only after reviewing related work.
