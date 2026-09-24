
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import re
import joblib
import tldextract
from urllib.parse import urlparse
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load models
text_model = joblib.load("/content/scholarship_scam_detection/scholarship_text_xgboost.pkl")
text_tfidf = joblib.load("/content/scholarship_scam_detection/scholarship_text_tfidf.pkl")
url_model = joblib.load("/content/scholarship_scam_detection/scholarship_url_xgboost.pkl")
tld_mapping = joblib.load("/content/scholarship_scam_detection/scholarship_tld_mapping.pkl")

# Text preprocessing
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


# URL feature extraction
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

    special_char_count = len(
        re.findall(r"[^a-zA-Z0-9]", url)
    )

    digits_count = len(
        re.findall(r"\d", url)
    )

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


# URL prediction
def predict_url(url):

    features = extract_url_features(url)

    # Use EXACT mapping from training
    tld = features["tld"]

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

    prediction = url_model.predict(input_data)[0]
    probability = url_model.predict_proba(input_data)[0]

    confidence = float(max(probability) * 100)

    if prediction == 0:
        result = "Legitimate"
        reason = "The URL does not show strong phishing-related characteristics."
    else:
        result = "Phishing"
        reason = "The URL contains characteristics associated with suspicious or phishing URLs."

    return result, confidence, reason


# Text prediction
def predict_text(text):

    cleaned_text = preprocess_text(text)

    text_vector = text_tfidf.transform([cleaned_text])

    prediction = text_model.predict(text_vector)[0]
    probability = text_model.predict_proba(text_vector)[0]

    confidence = float(max(probability) * 100)

    if prediction == 0:
        result = "Genuine"
        reason = "The message does not show strong scam-related patterns."
    else:
        result = "Scam"
        reason = "The message contains patterns associated with scholarship scam messages."

    return result, confidence, reason


# Home page
@app.route("/")
def home():
    return jsonify({"message": "Scholarship Scam Detection API is running."})


# Text API
@app.route("/predict_text", methods=["POST"])
def predict_text_api():

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({
            "error": "Please provide scholarship text."
        }), 400

    text = data["text"]

    if not text.strip():
        return jsonify({
            "error": "Text input cannot be empty."
        }), 400

    result, confidence, reason = predict_text(text)

    scam_type = get_text_scam_type(text)
    explanation = get_text_explanation(text)

    return jsonify({
        "input_type": "Text",
        "prediction": result,
        "confidence": round(confidence, 2),
        "scam_type": scam_type,
        "url_signals": [],
        "explanation": explanation
    })


# URL API
@app.route("/predict_url", methods=["POST"])
def predict_url_api():

    data = request.get_json()

    if not data or "url" not in data:
        return jsonify({"error": "Please provide a URL."}), 400

    url = data["url"].strip()

    if not url:
        return jsonify({"error": "URL input cannot be empty."}), 400

    result, confidence, reason = predict_url(url)

    return jsonify({
        "input_type": "URL",
        "prediction": result,
        "confidence": round(confidence, 2),
        "reason": reason
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
