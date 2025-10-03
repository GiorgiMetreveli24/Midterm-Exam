# Pearson Correlation and Scatter Plot

The first assignment is stored in file Finding-the-correlation.Where **Pearson's correlation coefficient** is calculated for a given set of data points and visualizes the relationship with a scatter plot and regression line.

---

## 📌 Features

- Calculates **Pearson's correlation coefficient** using NumPy.
- Generates a **scatter plot** of the given data.
- Adds a **regression line** to the scatter plot.
- Saves the visualization as `correlation_graph.png`.

---

## 📂 Files

- `solution.py` → The main Python script containing the implementation.
- `correlation_graph.png` → The generated graph showing the data points and regression line (created after running the script).

---

## ▶️ How to Run

1. Make sure you have **Python 3.x** installed.
2. Install the required libraries:
   ```bash
   pip install numpy matplotlib
   ```
3. ```bash
   python Finding-the-correlation/solution.py
   ```

# Spam Email Detection — Midterm

**Course:** AI and ML for Cybersecurity  
**Date:** October 3, 2025

## 1) Data file (uploaded)

Original dataset used in this project is available locally at `/mnt/data/g_metreveli2024_127645.csv`.  
Include this file in your repository under `data/` and reference it in the code.

## 2) Model training (70%) and source code

We trained a **Logistic Regression** model using a 70/30 stratified split.

- Source code: [`spam_detector.py`](spam_detector.py)
- Training: `python spam_detector.py train --data data/g_metreveli2024_127645.csv --out artifacts`

### Data loading & processing (summary)

- Load CSV with pandas
- Split features **[words, links, capital_words, spam_word_count]** from label **is_spam**
- Perform `train_test_split` with `test_size=0.30`, `random_state=42`, `stratify=y`
- Train `LogisticRegression(max_iter=1000, solver="liblinear")`

### Logistic regression model used

Scikit-learn Logistic Regression (binary), default L2 regularization.

### Learned coefficients

- Intercept: **-7.806318**
- Coefficients (order = words, links, capital_words, spam_word_count):

| Feature         | Coefficient |
| --------------- | ----------- |
| words           | 0.006495    |
| links           | 0.658211    |
| capital_words   | 0.380789    |
| spam_word_count | 0.645185    |

## 3) Test-set evaluation (30% held-out)

- **Accuracy:** **0.9653**
- **Confusion Matrix** (rows = true [0,1], cols = predicted [0,1]):  
  `[[371, 8], [18, 353]]`

Code used: `sklearn.metrics.accuracy_score` and `confusion_matrix`.

## 4) Email text parsing & feature extraction

The app can parse raw email text and extract the exact 4 features present in the dataset:

- **words**: word count (`[A-Za-z']+` tokens)
- **links**: number of URLs (`http(s)://` or `www.` pattern)
- **capital_words**: count of words in full uppercase (length > 1)
- **spam_word_count**: total occurrences of common spam indicators (e.g., _free, winner, prize, offer, click, bonus, bitcoin,..._).  
  Prediction command:

```
python spam_detector.py predict-text --model artifacts/logreg_model.joblib --text "your email here"
```

## 5) Example email that the model classifies as **spam**

```
CONGRATULATIONS WINNER! You have been selected for a LIMITED OFFER.
Click the link NOW to claim your FREE prize and BONUS cash:
https://example-giveaway.com/claim https://promo.example.com
ACT NOW! This is URGENT and will expire soon.
```

- Predicted: **spam (1)** with high probability.
- Rationale: multiple **links**, many **CAPITAL** words, and several **spam keywords**.

## 6) Example email that the model classifies as **legitimate**

```
Hi Nino,
Here are the meeting minutes from today. I attached the document and summarized action items.
Let's review next week and confirm the deployment plan.
Thanks,
Lasha
```

- Predicted: **legitimate (0)**.
- Rationale: normal phrasing, no links, no typical spam words, no all-caps shouting.

## 7) Visualizations

See the `plots/` folder:

- `class_distribution.png` — class balance bar chart.
- `feature_correlation_heatmap.png` — heatmap of feature correlations.

---

### Repro quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
mkdir -p data && cp /mnt/data/g_metreveli2024_127645.csv data/
python spam_detector.py train --data data/g_metreveli2024_127645.csv --out artifacts
python spam_detector.py predict-text --model artifacts/logreg_model.joblib --text "Your email text here"
```
