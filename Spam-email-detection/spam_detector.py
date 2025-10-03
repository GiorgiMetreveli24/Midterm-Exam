#!/usr/bin/env python3
import argparse, re, joblib, numpy as np, pandas as pd, sys, os
from pathlib import Path

# === Feature extractor ===
SPAM_WORDS = [
    "free","winner","prize","credit","loan","urgent","limited","offer","click",
    "buy","discount","deal","bonus","viagra","casino","bet","bitcoin","crypto",
    "money","cash","win","guarantee","act now","no cost","cheap","urgent","gift"
]

URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)

def extract_features_from_text(text: str):
    words = re.findall(r"[A-Za-z']+", text)
    word_count = len(words)
    links = len(URL_RE.findall(text))
    capital_words = sum(1 for w in words if len(w) > 1 and w.isupper())
    lower_text = text.lower()
    spam_word_count = 0
    for w in SPAM_WORDS:
        spam_word_count += lower_text.count(w)
    return {"words": word_count, "links": links, "capital_words": capital_words, "spam_word_count": spam_word_count}

def as_row(feats: dict):
    return np.array([[feats["words"], feats["links"], feats["capital_words"], feats["spam_word_count"]]], dtype=float)

def train(data_csv: str, out_dir: str):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score

    df = pd.read_csv(data_csv)
    X = df.drop(columns=["is_spam"])
    y = df["is_spam"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)

    # Save artifacts
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "logreg_model.joblib")

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("Accuracy:", acc)
    print("Confusion Matrix:", cm)
    print("Coefficients:", model.coef_.tolist())
    print("Intercept:", model.intercept_.tolist())

def evaluate(data_csv: str, model_path: str):
    import pandas as pd
    from sklearn.metrics import confusion_matrix, accuracy_score

    model = joblib.load(model_path)
    df = pd.read_csv(data_csv)
    X = df.drop(columns=["is_spam"])
    y = df["is_spam"]

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred).tolist()

    print("Full-dataset evaluation")
    print("Accuracy:", acc)
    print("Confusion Matrix:", cm)

def predict_text(model_path: str, text: str):
    model = joblib.load(model_path)
    feats = extract_features_from_text(text)
    X = as_row(feats)
    prob = float(model.predict_proba(X)[0,1])
    pred = int(prob >= 0.5)
    print("Features:", feats)
    print("Predicted class:", pred, "(1=spam, 0=legit)")
    print("Spam probability:", prob)

def main():
    p = argparse.ArgumentParser(description="Email spam detector (logistic regression).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train model on CSV")
    p_train.add_argument("--data", required=True, help="Path to CSV with columns words,links,capital_words,spam_word_count,is_spam")
    p_train.add_argument("--out", default="artifacts", help="Output directory for model artifacts")

    p_eval = sub.add_parser("eval", help="Evaluate an existing model on a CSV")
    p_eval.add_argument("--data", required=True)
    p_eval.add_argument("--model", required=True)

    p_pred = sub.add_parser("predict-text", help="Predict class for a raw email text")
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--text", required=True, help="Email text to classify (quote it)")

    args = p.parse_args()
    if args.cmd == "train":
        train(args.data, args.out)
    elif args.cmd == "eval":
        evaluate(args.data, args.model)
    elif args.cmd == "predict-text":
        predict_text(args.model, args.text)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
