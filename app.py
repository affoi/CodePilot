# =========================================
# FINAL APP.PY
# COMPLETE VERSION
# =========================================

import streamlit as st
import numpy as np
import re
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
# BACKGROUND
# =========================================

def add_bg_from_local(image_file):

    with open(image_file, "rb") as image:

        encoded = base64.b64encode(
            image.read()
        ).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image: url(
            "data:image/png;base64,{encoded}"
            );
            background-size: cover;
            background-position: center;
        }}

        .stTextArea textarea {{
            background-color: white !important;
            color: black !important;
        }}

        .stTextInput input {{
            background-color: white !important;
            color: black !important;
        }}
        h1 {{
            color: #FFF8E7 !important;
        }}
        h2, h3, p, li, div, span {{
            color: white !important;
        }}

        .stMarkdown {{
            color: white !important;
        }}

        .stText {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("background.png")

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

        # Detect loops
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
# CONSTRAINT PARSER
# =========================================

def parse_constraint(text):

    text = str(text).lower()

    text = text.replace(" ", "")

    if "<=" in text:

        expr = text.split("<=")[-1]

    else:

        expr = text

    expr = expr.replace("^", "**")

    allowed = "0123456789*+-/()"

    for ch in expr:

        if ch not in allowed:
            raise ValueError

    return int(eval(expr))

# =========================================
# RULE ENGINE
# =========================================

def rule_based_prediction(code):

    code = str(code).lower()

    depth = max_loop_depth(code)

    _, recursive_calls = (
        get_function_name_and_calls(code)
    )

    loop_patterns = re.findall(
        r'for\s*\(.*?\)|while\s*\(.*?\)',
        code,
        re.DOTALL
    )

    loop_count = len(loop_patterns)

    # O(n log n)
    if re.search(
        r'\barrays\.sort\s*\(|'
        r'\bcollections\.sort\s*\(',
        code
    ):

        return "O(n log n)"

    # O(log n)
    if (
        ("while" in code or "for" in code)
        and (
            "/ 2" in code
            or "mid" in code
            or ">>" in code
        )
    ):

        return "O(log n)"

    # O(2^n)
    if recursive_calls >= 2:

        return "O(2^n)"

    # O(n^3)
    if depth >= 3 or loop_count >= 3:

        return "O(n^3)"

    # O(n^2)
    if depth == 2 or loop_count == 2:

        return "O(n^2)"

    # O(n)
    if depth == 1 or loop_count == 1:

        return "O(n)"

    # O(1)
    return "O(1)"

# =========================================
# TLE PREDICTION
# =========================================

def tle_prediction(complexity, n):

    if complexity in ["O(1)", "O(log n)"]:
        return "LOW"

    if complexity == "O(n)":

        if n <= 10**7:
            return "LOW"

        return "MEDIUM"

    if complexity == "O(n log n)":

        if n <= 10**6:
            return "LOW"

        return "MEDIUM"

    if complexity == "O(n^2)":

        if n <= 10**4:
            return "MEDIUM"

        return "HIGH"

    if complexity in ["O(n^3)", "O(2^n)"]:
        return "HIGH"

    return "UNKNOWN"

# =========================================
# OPTIMIZATION SUGGESTIONS
# =========================================

def optimization_suggestions(complexity):

    if complexity == "O(1)":

        return [
            "Excellent constant time solution",
            "Highly optimized implementation",
            "Very scalable for large inputs"
        ]

    if complexity == "O(log n)":

        return [
            "Excellent optimization",
            "Binary-search style solution detected",
            "Highly scalable approach",
            "Ideal for very large constraints"
        ]

    if complexity == "O(n)":

        return [
            "Efficient linear solution",
            "Good for large constraints",
            "Consider hashing for faster lookup",
            "Can be optimized further with preprocessing"
        ]

    if complexity == "O(n log n)":

        return [
            "Efficient divide-and-conquer solution",
            "Good scalability",
            "Suitable for competitive programming",
            "Efficient sorting/searching approach"
        ]

    if complexity == "O(n^2)":

        return [
            "Use HashMap for constant time lookup",
            "Try two-pointer approach",
            "Use sliding window technique",
            "Reduce nested loops",
            "Consider binary search optimization",
            "Use prefix sums if applicable"
        ]

    if complexity == "O(n^3)":

        return [
            "Triple nested loops detected",
            "Apply dynamic programming",
            "Optimize matrix operations",
            "Reduce redundant computations",
            "Use preprocessing",
            "Use memoization"
        ]

    if complexity == "O(2^n)":

        return [
            "Exponential recursion detected",
            "Use memoization",
            "Apply dynamic programming",
            "Avoid repeated recursive states",
            "Convert recursion to iterative DP",
            "Use pruning techniques"
        ]

    return [
        "Complexity could not be fully analyzed"
    ]

# =========================================
# LOAD MODEL
# =========================================

model = joblib.load(
    "saved_model.pkl"
)

vectorizer = joblib.load(
    "vectorizer.pkl"
)

encoder = joblib.load(
    "encoder.pkl"
)

# =========================================
# UI
# =========================================

st.title(
    "CodePilot - Complexity Analyzer"
)


code = st.text_area(
    "Paste your Java code",
    height=350
)

constraint = st.text_input(
    "Constraints",
    value="1 <= n <= 10^5"
)

# =========================================
# ANALYZE
# =========================================

if st.button("Analyze"):

    try:

        parsed_n = parse_constraint(
            constraint
        )

    except:

        st.error(
            "Invalid constraint format"
        )

        st.stop()

    struct = np.array(
        extract_features(code)
    ).reshape(1, -1)

    text = vectorizer.transform(
        [code]
    ).toarray()

    final = np.hstack((
        struct,
        text
    ))

    pred_encoded = model.predict(
        final
    )[0]

    pred = encoder.inverse_transform(
        [pred_encoded]
    )[0]

    final_pred = (
        rule_based_prediction(code)
    )

    confidence = round(
        np.max(
            model.predict_proba(final)[0]
        ) * 100,
        2
    )

    tle_risk = tle_prediction(
        final_pred,
        parsed_n
    )

    suggestions = optimization_suggestions(
        final_pred
    )

    st.success(
        f"Complexity: {final_pred}"
    )

    st.write(
        f"Confidence: {confidence}%"
    )

    st.write(
        f"Parsed Constraint: {parsed_n}"
    )

    st.subheader(
        "TLE Risk"
    )

    st.write(tle_risk)

    st.subheader(
        "Optimization Suggestions"
    )

    for s in suggestions:

        st.write(f"- {s}")