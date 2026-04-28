import pandas as pd
import numpy as np
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt


# =========================================
# LOAD DATA
# =========================================

df = pd.read_csv("final_cleaned_dataset.csv")

df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(str).str.strip()
df = df[df["label"] != ""]


# =========================================
# HELPERS
# =========================================

def max_loop_depth(code):
    lines = str(code).split("\n")
    depth = 0
    stack = []

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if "for" in stripped or "while" in stripped:
            while stack and stack[-1] >= indent:
                stack.pop()

            stack.append(indent)
            depth = max(depth, len(stack))

    return depth


def get_function_name_and_calls(code):
    code = str(code)

    # Better Java function detection
    match = re.search(
        r'(?:public|private|protected)?\s*'
        r'(?:static\s+)?'
        r'(?:int|void|double|float|String|boolean)\s+'
        r'(\w+)\s*\(',
        code
    )

    name = match.group(1) if match else ""

    if not name:
        return "", 0

    calls = len(
        re.findall(
            r'\b' + re.escape(name) + r'\s*\(',
            code
        )
    ) - 1

    return name, calls


def extract_features(code):
    code = str(code).lower()

    loops = code.count("for") + code.count("while")
    depth = max_loop_depth(code)

    has_sort = 1 if "sort(" in code else 0
    has_divide = 1 if "mid" in code or "/ 2" in code else 0

    _, recursive_calls = get_function_name_and_calls(code)

    has_recursion = 1 if recursive_calls > 0 else 0

    return [
        loops,
        depth,
        has_sort,
        has_divide,
        has_recursion,
        recursive_calls
    ]


# =========================================
# FEATURE ENGINEERING
# =========================================

df["features"] = df["code"].apply(extract_features)

X_struct = np.array(list(df["features"]))

vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2)
)

X_text = vectorizer.fit_transform(df["code"])

X = np.hstack((
    X_struct,
    X_text.toarray()
))

encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])


# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# =========================================
# MODELS
# =========================================

lr = LogisticRegression(
    max_iter=1000
)

rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

xgb = XGBClassifier(
    eval_metric="mlogloss"
)


# =========================================
# TRAINING
# =========================================

print("Training Logistic Regression...")
lr.fit(X_train, y_train)

print("Training Random Forest...")
rf.fit(X_train, y_train)

print("Training XGBoost...")
xgb.fit(X_train, y_train)


# =========================================
# EVALUATION FUNCTION
# =========================================

accuracy_results = []

def evaluate(name, model):
    pred_encoded = model.predict(X_test)

    pred = encoder.inverse_transform(pred_encoded)
    true = encoder.inverse_transform(y_test)

    acc = accuracy_score(true, pred)

    print(f"\n===== {name} =====")
    print("Accuracy:", round(acc * 100, 2), "%")
    print(classification_report(true, pred, zero_division=0))

    accuracy_results.append({
        "Model": name,
        "Accuracy (%)": round(acc * 100, 2)
    })

    return pred, true


# =========================================
# RUN EVALUATION
# =========================================

lr_pred, true_labels = evaluate(
    "Logistic Regression",
    lr
)

rf_pred, _ = evaluate(
    "Random Forest",
    rf
)

xgb_pred, _ = evaluate(
    "XGBoost",
    xgb
)


# =========================================
# ACCURACY COMPARISON TABLE
# =========================================

print("\n===================================")
print("ACCURACY COMPARISON TABLE")
print("===================================")

accuracy_df = pd.DataFrame(accuracy_results)

print(accuracy_df)

# Optional: Save to CSV for report use
accuracy_df.to_csv(
    "model_accuracy_comparison.csv",
    index=False
)

print("\nSaved as: model_accuracy_comparison.csv")


# =========================================
# CONFUSION MATRIX (Random Forest)
# =========================================

print("\nShowing Random Forest Confusion Matrix...")

cm = confusion_matrix(
    true_labels,
    rf_pred
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()