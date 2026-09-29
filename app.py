from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import joblib
import tldextract
from urllib.parse import urlparse
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

app = Flask(__name__)
CORS(app)  # Allow requests from the Netlify frontend

# -----------------------------------------------------------------------------
# Project paths - works whether app.py is run from /content or its own folder
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

text_model = joblib.load(BASE_DIR / "scholarship_text_xgboost.pkl")
text_tfidf = joblib.load(BASE_DIR / "scholarship_text_tfidf.pkl")
url_model = joblib.load(BASE_DIR / "scholarship_url_xgboost.pkl")
tld_mapping = joblib.load(BASE_DIR / "scholarship_tld_mapping.pkl")

# -----------------------------------------------------------------------------
# Text preprocessing - kept consistent with model training
# -----------------------------------------------------------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


# -----------------------------------------------------------------------------
# Simple explanation / scam-type helpers
# NOTE: These are explanation rules, not a separate ML model.
# -----------------------------------------------------------------------------
FEE_WORDS = [
    "fee", "payment", "pay", "transfer", "deposit", "charge",
    "handling fee", "registration fee", "processing fee"
]
OTP_BANK_WORDS = [
    "otp", "one time password", "password", "bank", "account",
    "card", "upi", "pin", "verify account"
]
URGENCY_WORDS = [
    "urgent", "immediately", "today", "now", "hurry", "limited time",
    "last chance", "act fast"
]
REWARD_WORDS = [
    "winner", "won", "prize", "reward", "selected", "claim",
    "double", "guaranteed"
]


def contains_any(text, words):
    text_lower = str(text).lower()
    return [word for word in words if word in text_lower]


def get_text_scam_details(text, prediction):
    if prediction == 0:
        return "None", "The message does not show strong scam-related patterns."

    fee = contains_any(text, FEE_WORDS)
    otp_bank = contains_any(text, OTP_BANK_WORDS)
    urgency = contains_any(text, URGENCY_WORDS)
    reward = contains_any(text, REWARD_WORDS)

    if fee:
        scam_type = "Fee / Payment Scam"
        explanation = "The message contains fee or payment-related wording that is commonly associated with scholarship scams."
    elif otp_bank:
        scam_type = "Account / Credential Scam"
        explanation = "The message contains account, password, OTP, banking, or payment-related wording that may indicate an attempt to obtain sensitive information."
    elif urgency:
        scam_type = "Urgency / Pressure Scam"
        explanation = "The message uses urgent or pressure-based wording that can be associated with fraudulent scholarship offers."
    elif reward:
        scam_type = "Fake Reward / Selection Scam"
        explanation = "The message contains reward, prize, selection, or guaranteed-benefit wording associated with suspicious scholarship offers."
    else:
        scam_type = "Scholarship Scam"
        explanation = "The message contains patterns associated with scholarship scam messages."

    return scam_type, explanation


# -----------------------------------------------------------------------------
# URL feature extraction - kept consistent with model training
# -----------------------------------------------------------------------------
def extract_url_features(url):
    url = str(url).strip()
    parsed = urlparse(url)
    extracted = tldextract.extract(url)
    tld = extracted.suffix
    url_lower = url.lower()

    suspicious_words_list = [
        "login", "verify", "verification", "account",
        "update", "secure", "security", "confirm",
        "password", "bank", "payment", "free",
        "winner", "prize", "claim", "urgent",
        "scholarship", "fee", "apply"
    ]

    suspicious_words = sum(
        1 for word in suspicious_words_list
        if word in url_lower
    )

    ip_pattern = r"^(?:https?://)?(?:\d{1,3}\.){3}\d{1,3}"
    has_ip = 1 if re.search(ip_pattern, url) else 0
    has_https = 1 if parsed.scheme.lower() == "https" else 0
    url_length = len(url)
    num_dots = url.count(".")
    num_subdirs = url.count("/")
    num_params = url.count("?") + url.count("&")
    special_char_count = len(re.findall(r"[^a-zA-Z0-9]", url))
    digits_count = len(re.findall(r"\d", url))

    if len(url) > 0:
        probabilities = [
            url.count(char) / len(url)
            for char in set(url)
        ]
        entropy = -sum(
            p * np.log2(p)
            for p in probabilities
        )
    else:
        entropy = 0

    return {
        "url_length": url_length,
        "num_dots": num_dots,
        "has_https": has_https,
        "has_ip": has_ip,
        "num_subdirs": num_subdirs,
        "num_params": num_params,
        "suspicious_words": suspicious_words,
        "tld": tld,
        "special_char_count": special_char_count,
        "digits_count": digits_count,
        "entropy": entropy
    }


def get_url_details(features, prediction):
    signals = []

    if features["has_https"] == 0:
        signals.append("No HTTPS")
    if features["has_ip"] == 1:
        signals.append("IP address used")
    if features["suspicious_words"] > 0:
        signals.append("Suspicious URL words")
    if features["url_length"] > 75:
        signals.append("Long URL")
    if features["num_params"] > 2:
        signals.append("Multiple URL parameters")
    if features["num_subdirs"] > 4:
        signals.append("Many subdirectories")
    if features["digits_count"] > 5:
        signals.append("Many digits")

    if prediction == 0:
        scam_type = "None"
        explanation = "The URL does not show strong phishing-related characteristics."
        if not signals:
            signals = ["No major suspicious URL signals detected"]
    else:
        scam_type = "Phishing / Suspicious Scholarship URL"
        if signals:
            explanation = "The URL contains characteristics associated with suspicious or phishing URLs."
        else:
            explanation = "The model classified the URL as phishing based on the learned URL patterns."

    return scam_type, explanation, signals

def predict_url(url):
    features = extract_url_features(url)
    tld = features["tld"]

    # Known legitimate domains
    trusted_domains = {
        "scholarships.gov.in",
        "education.gov.in",
        "india.gov.in",
        "toyota.com",
        "quora.com",
        "trustedsite58.org",
        "trustedsite73.org",
        "newsportal58.com",
        "newsportal108.com",
        "newsportal23.com",
        "newsportal91.com",
        "newsportal49.com",
        "university85.edu",
        "university83.edu",
        "university104.edu"
    }

    parsed = urlparse(
        url if "://" in url else "https://" + url
    )

    domain = parsed.netloc.lower().split(":")[0]

    if domain.startswith("www."):
        domain = domain[4:]

    # Trusted-domain validation
    if domain in trusted_domains:
        result = "Legitimate"
        confidence = 99.0
        scam_type = "No Scam Detected"
        explanation = "The domain matches a known legitimate domain."
        signals = ["Trusted domain"]

        return result, confidence, scam_type, explanation, signals

    # Normal XGBoost prediction
    if tld in tld_mapping:
        features["tld"] = tld_mapping[tld]
    else:
        features["tld"] = tld_mapping.get("unknown", 0)

    feature_order = [
        "url_length",
        "num_dots",
        "has_https",
        "has_ip",
        "num_subdirs",
        "num_params",
        "suspicious_words",
        "tld",
        "special_char_count",
        "digits_count",
        "entropy"
    ]

    input_data = pd.DataFrame(
        [[features[col] for col in feature_order]],
        columns=feature_order
    )

    prediction = int(url_model.predict(input_data)[0])
    probability = url_model.predict_proba(input_data)[0]
    confidence = float(max(probability) * 100)

    result = "Legitimate" if prediction == 0 else "Phishing"

    scam_type, explanation, signals = get_url_details(
        features,
        prediction
    )

    return result, confidence, scam_type, explanation, signals


def predict_text(text):
    cleaned_text = preprocess_text(text)
    text_vector = text_tfidf.transform([cleaned_text])
    prediction = int(text_model.predict(text_vector)[0])
    probability = text_model.predict_proba(text_vector)[0]
    confidence = float(max(probability) * 100)

    result = "Genuine" if prediction == 0 else "Scam"
    scam_type, explanation = get_text_scam_details(text, prediction)

    return result, confidence, scam_type, explanation


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def home():
    # The Netlify frontend is used for the UI. This route is only a backend check.
    return jsonify({
        "status": "Backend is running",
        "message": "Scholarship Scam Detection API"
    })


@app.route("/predict_text", methods=["POST"])
def predict_text_api():
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Please provide scholarship text."}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": "Text input cannot be empty."}), 400

    result, confidence, scam_type, explanation = predict_text(text)

    return jsonify({
        "input_type": "Text",
        "prediction": result,
        "confidence": round(confidence, 2),
        "scam_type": scam_type,
        "explanation": explanation,
        "url_signals": []
    })


@app.route("/predict_url", methods=["POST"])
def predict_url_api():
    data = request.get_json(silent=True)

    if not data or "url" not in data:
        return jsonify({"error": "Please provide a URL."}), 400

    url = str(data["url"]).strip()
    if not url:
        return jsonify({"error": "URL input cannot be empty."}), 400

    result, confidence, scam_type, explanation, url_signals = predict_url(url)

    return jsonify({
        "input_type": "URL",
        "prediction": result,
        "confidence": round(confidence, 2),
        "scam_type": scam_type,
        "explanation": explanation,
        "url_signals": url_signals
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
