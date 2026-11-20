"""Train and compare the models from the command line.

    python train.py                                   # default dataset (airline tweets)
    python train.py --csv my.csv --text-col review --label-col sentiment

Saves models/naive_bayes.joblib, models/logistic_regression.joblib,
models/metrics.json and models/test_predictions.csv (used by the dashboard).
"""
import argparse

from sentiment.model import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", help="your own labelled CSV (default: Twitter US Airline Sentiment)")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    run_training(args.csv, args.text_col, args.label_col, args.test_size, args.out)


if __name__ == "__main__":
    main()
