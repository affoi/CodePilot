# =========================================
# READY TO RUN FRONTEND CODE (UPDATED)
# Includes:
# ✅ Better recursion detection
# ✅ Fibonacci → O(2^n)
# ✅ Improved TLE prediction
# ✅ number_input() instead of text_input()
# ✅ No more "Invalid" issue
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


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

    match = re.search(
        r'(?:public|private|protected)?\s*(?:static\s+)?(?:int|void|double|float|String|boolean)\s+(\w+)\s*\(',
        code
    )

    name = match.group(1) if match else ""

    if not name:
        return "", 0

    calls = len(
        re.findall(r'\b' + re.escape(name) + r'\s*\(', code)
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
# RULE BASED PREDICTION
# =========================================

def rule_based_prediction(code):
    code = str(code).lower()
    depth = max_loop_depth(code)

    _, recursive_calls = get_function_name_and_calls(code)

    if "sort(" in code:
        return "O(n log n)"

    if "/ 2" in code or ">>" in code:
        return "O(log n)"

    if recursive_calls >= 2:
        return "O(2^n)"

    if recursive_calls == 1:
        return "O(n)"

    if depth >= 3:
        return "O(n^3)"

    if depth == 2:
        return "O(n^2)"

    if depth == 1:
        return "O(n)"

    return "O(1)"


# =========================================
# IMPROVED TLE PREDICTION
# =========================================

def tle_prediction(complexity, n):

    if not str(n).strip():
        return "NOT PROVIDED", "Please enter input constraint n"

    try:
        n = int(n)
    except:
        return "INVALID INPUT", "Enter only numeric value like 100000"

    if complexity in ["O(1)", "O(log n)"]:
        return "LOW", "Very safe"

    if complexity == "O(n)":
        if n <= 10**7:
            return "LOW", "Usually acceptable"
        else:
            return "MEDIUM", "Large input size"

    if complexity == "O(n log n)":
        if n <= 10**6:
            return "LOW", "Efficient"
        else:
            return "MEDIUM", "May be heavy"

    if complexity == "O(n^2)":
        if n <= 10**4:
            return "MEDIUM", "Borderline case"
        else:
            return "HIGH", "Likely TLE"

    if complexity in ["O(n^3)", "O(2^n)"]:
        return "HIGH", "Very likely TLE"

    return "UNKNOWN", "Unable to determine"


# =========================================
# OPTIMIZATION SUGGESTIONS
# =========================================

def optimization_suggestions(complexity):
    if complexity == "O(n^2)":
        return [
            "Use HashMap",
            "Try two-pointer or prefix sum"
        ]

    if complexity == "O(n^3)":
        return [
            "Reduce nested loops",
            "Use DP or preprocessing"
        ]

    if complexity == "O(2^n)":
        return [
            "Use memoization",
            "Try dynamic programming"
        ]

    return ["Current approach looks efficient"]


# =========================================
# USER HISTORY
# =========================================

def update_behavior(complexity):
    file = "user_behavior.json"
    data = {}

    if os.path.exists(file):
        with open(file, "r") as f:
            data = json.load(f)

    data[complexity] = data.get(complexity, 0) + 1

    with open(file, "w") as f:
        json.dump(data, f)

    return data


def personalized_feedback(data):
    if not data:
        return "No history yet"

    common = max(data, key=data.get)

    if common in ["O(n^2)", "O(n^3)"]:
        return "You often use brute-force solutions. Try optimized approaches like hashing or binary search."

    return "Good complexity patterns overall."


# =========================================
# TRAIN MODEL
# =========================================

@st.cache_resource
def train_model():
    df = pd.read_csv("final_cleaned_dataset.csv")

    df = df.dropna(subset=["label"])

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])

    df["features"] = df["code"].apply(extract_features)
    X_struct = np.array(list(df["features"]))

    vectorizer = TfidfVectorizer(max_features=1000)
    X_text = vectorizer.fit_transform(df["code"])

    X = np.hstack((X_struct, X_text.toarray()))

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, vectorizer, encoder


model, vectorizer, encoder = train_model()


# =========================================
# UI
# =========================================

st.title("Personalized Coding Assistant")

code = st.text_area(
    "Paste your code",
    height=300
)

# CHANGED HERE ↓↓↓
constraint = st.number_input(
    "Enter constraint n",
    min_value=1,
    value=1000,
    step=1
)

if st.button("Analyze"):

    struct = np.array(
        extract_features(code)
    ).reshape(1, -1)

    text = vectorizer.transform(
        [code]
    ).toarray()

    final = np.hstack((struct, text))

    pred_encoded = model.predict(final)[0]

    pred = encoder.inverse_transform(
        [pred_encoded]
    )[0]

    confidence = round(
        np.max(model.predict_proba(final)[0]) * 100,
        2
    )

    rule = rule_based_prediction(code)

    final_pred = rule if rule != "O(1)" else pred

    tle_risk, tle_reason = tle_prediction(
        final_pred,
        constraint
    )

    suggestions = optimization_suggestions(
        final_pred
    )

    behavior = update_behavior(
        final_pred
    )

    feedback = personalized_feedback(
        behavior
    )

    st.success(f"Complexity: {final_pred}")
    st.write(f"Confidence: {confidence}%")

    st.subheader("TLE Prediction")
    st.write(f"Risk: {tle_risk}")
    st.write(tle_reason)

    st.subheader("Optimization Suggestions")
    for s in suggestions:
        st.write(f"- {s}")

    st.subheader("Personalized Feedback")
    st.write(feedback)