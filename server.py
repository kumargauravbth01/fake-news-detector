import re
import joblib
import numpy as np
import nltk
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# NLTK setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

app = Flask(__name__)
CORS(app)

# Load saved ML model & vectorizer
model = joblib.load('model.pkl')
tfidf = joblib.load('vectorizer.pkl')

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

# Manifest route - PWABuilder / PWA ke liye
@app.route('/manifest.json')
def manifest():
    return send_from_directory('.', 'manifest.json')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    news_text = data.get('text', '').strip()

    if not news_text:
        return jsonify({'error': 'No text provided'}), 400

    cleaned = clean_text(news_text)
    vec = tfidf.transform([cleaned])
    prediction = int(model.predict(vec)[0])

    # Calculate confidence score
    score = model.decision_function(vec)[0]
    confidence = float(1 / (1 + np.exp(-abs(score))) * 100)

    result = {
        'prediction': 'REAL' if prediction == 1 else 'FAKE',
        'confidence': round(confidence, 2)
    }
    return jsonify(result)

if __name__ == '__main__':
    # host='0.0.0.0' allows access from mobile devices on the same Wi-Fi
    app.run(host='0.0.0.0', port=5000, debug=True)