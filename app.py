# =========================================
# FRONTEND CODE (app.py)
# DEPLOYMENT READY FOR GCP
# DARK BACKGROUND UI VERSION
# Uses joblib.load()
# Background Image + Better Text Visibility
# =========================================

import streamlit as st
import numpy as np
import re
import json
import os
import joblib
import base64

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="CodePilot",
    page_icon="💻",
    layout="centered"
)

# =========================================
# BACKGROUND IMAGE + UI STYLING
# =========================================

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Main content box */
        .main {{
            background-color: rgba(0, 0, 0, 0.55);
            border-radius: 15px;
            padding: 20px;
        }}

        /* Title */
        h1 {{
            color: #FFFFFF !important;
            font-weight: 700;
        }}

        /* Subheaders */
        h2, h3 {{
            color: #EAF6FF !important;
        }}

        /* Normal text */
        p, label, div, span {{
            color: #F5F5F5 !important;
            font-size: 16px;
        }}

        /* Text area */
        textarea {{
            background-color: rgba(255,255,255,0.92) !important;
            color: black !important;
            border-radius: 10px !important;
        }}

        /* Number input */
        input {{
            background-color: rgba(255,255,255,0.92) !important;
            color: black !important;
            border-radius: 10px !important;
        }}

        /* Button */
        .stButton > button {{
            background-color: #00B4D8;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 24px;
            font-weight: bold;
        }}

        .stButton > button:hover {{
            background-color: #0096C7;
            color: white;
        }}

        /* Success box */
        .stSuccess {{
            background-color: rgba(0, 255, 150, 0.15) !important;
            color: white !important;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Put your image file in same folder as app.py
add_bg_from_local("background.png")

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
        r'(?:public|private|protected)?\s*(?:static\s+)?'
        r'(?:int|void|double|float|String|boolean)\s+'
        r'(\w+)\s*\(',
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
# TLE PREDICTION
# =========================================

def tle_prediction(complexity, n):
    n = int(n)

    if complexity in ["O(1)", "O(log n)"]:
        return "LOW", "Very safe"

    if complexity == "O(n)":
        return (
            ("LOW", "Usually acceptable")
            if n <= 10**7 else
            ("MEDIUM", "Large input size")
        )

    if complexity == "O(n log n)":
        return (
            ("LOW", "Efficient")
            if n <= 10**6 else
            ("MEDIUM", "May be heavy")
        )

    if complexity == "O(n^2)":
        return (
            ("MEDIUM", "Borderline case")
            if n <= 10**4 else
            ("HIGH", "Likely TLE")
        )

    if complexity in ["O(n^3)", "O(2^n)"]:
        return "HIGH", "Very likely TLE"

    return "UNKNOWN", "Unable to determine"


# =========================================
# SUGGESTIONS
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
        return "You often use brute-force solutions. Try optimized approaches."

    return "Good complexity patterns overall."


# =========================================
# LOAD SAVED MODEL
# =========================================

model = joblib.load("saved_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
encoder = joblib.load("encoder.pkl")


# =========================================
# UI
# =========================================

st.title("CodePilot - Personalized Coding Assistant")
st.write("Final Model Used: Random Forest Classifier")

code = st.text_area(
    "Paste your code",
    height=300
)

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

    suggestions = optimization_suggestions(final_pred)

    behavior = update_behavior(final_pred)
    feedback = personalized_feedback(behavior)

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