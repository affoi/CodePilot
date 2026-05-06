# =========================================
# TRAINING CODE (FINAL IMPROVED VERSION)
# =========================================

import pandas as pd
import numpy as np
import re
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.preprocessing import (
    LabelEncoder
)

import seaborn as sns
import matplotlib.pyplot as plt

# =========================================
# LOAD DATA
# =========================================

df = pd.read_csv(
    "final_cleaned_dataset.csv"
)

df = df.dropna(subset=["label"])

df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
)

df = df[df["label"] != ""]

# =========================================
# LOOP DEPTH DETECTION
# =========================================

def max_loop_depth(code):

    lines = str(code).splitlines()

    depth = 0

    max_depth = 0

    stack = []

    for line in lines:

        stripped = line.strip()

        # Detect loop start
        if re.match(
            r'^(for|while)\s*\(',
            stripped
        ):

            depth += 1

            stack.append(depth)

            max_depth = max(
                max_depth,
                depth
            )

        # Detect closing braces
        if "}" in stripped and stack:

            depth -= stripped.count("}")

            if depth < 0:
                depth = 0

    return max_depth

# =========================================
# RECURSION DETECTION
# =========================================

def get_function_name_and_calls(code):

    code = str(code)

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

# =========================================
# FEATURE EXTRACTION
# =========================================

def extract_features(code):

    code = str(code).lower()

    loops = (
        code.count("for")
        + code.count("while")
    )

    depth = max_loop_depth(code)

    has_sort = (
        1 if re.search(
            r'\barrays\.sort\s*\(|'
            r'\bcollections\.sort\s*\(',
            code
        )
        else 0
    )

    has_divide = (
        1 if (
            "mid" in code
            or "/ 2" in code
            or ">>" in code
        )
        else 0
    )

    _, recursive_calls = (
        get_function_name_and_calls(code)
    )

    has_recursion = (
        1 if recursive_calls > 0 else 0
    )

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

df["features"] = df["code"].apply(
    extract_features
)

X_struct = np.array(
    list(df["features"])
)

vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2)
)

X_text = vectorizer.fit_transform(
    df["code"]
)

X = np.hstack((
    X_struct,
    X_text.toarray()
))

encoder = LabelEncoder()

y = encoder.fit_transform(
    df["label"]
)

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
)

# =========================================
# MODEL
# =========================================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    class_weight="balanced"
)

# =========================================
# TRAINING
# =========================================

print("Training Random Forest...")

rf.fit(X_train, y_train)

# =========================================
# EVALUATION
# =========================================

pred_encoded = rf.predict(X_test)

pred = encoder.inverse_transform(
    pred_encoded
)

true = encoder.inverse_transform(
    y_test
)

acc = accuracy_score(true, pred)

print("\n===== RANDOM FOREST =====")

print(
    "Accuracy:",
    round(acc * 100, 2),
    "%"
)

print(
    classification_report(
        true,
        pred,
        zero_division=0
    )
)

# =========================================
# SAVE FILES
# =========================================

joblib.dump(
    rf,
    "saved_model.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)

joblib.dump(
    encoder,
    "encoder.pkl"
)

print("\nSaved:")
print("saved_model.pkl")
print("vectorizer.pkl")
print("encoder.pkl")

# =========================================
# CONFUSION MATRIX
# =========================================

cm = confusion_matrix(
    true,
    pred
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()