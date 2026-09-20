import re
import os
import joblib
import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Page layout configuration
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Cache stop words and model to prevent reloading on every run
@st.cache_resource
def load_resources():
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    
    if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
        return None, None, stop_words, ps
        
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer, stop_words, ps

model, tfidf, stop_words, ps = load_resources()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Sidebar with project details for Viva/Presentation
with st.sidebar:
    st.header("📌 Project Details")
    st.markdown("""
    * **Domain:** Natural Language Processing (NLP)
    * **Algorithms Tested:** Passive Aggressive, Logistic Regression, Naive Bayes
    * **Feature Extractor:** TF-IDF Vectorizer (5,000 features, Unigrams + Bigrams)
    """)
    st.markdown("---")
    st.subheader("📊 Performance Visuals")
    if os.path.exists('model_comparison.png'):
        st.image('model_comparison.png', caption='Model Accuracy Comparison')
    if os.path.exists('confusion_matrix.png'):
        st.image('confusion_matrix.png', caption='Confusion Matrix')

# Main Screen
st.title("📰 AI-Powered Fake News Detection System")
st.write("Analyze online news headlines and articles in real-time to check credibility using machine learning.")

if model is None or tfidf is None:
    st.error("Model files (`model.pkl` ya `vectorizer.pkl`) missing hain! Kripya pehle terminal me `python train_model.py` run karein.")
    st.stop()

# Text input
news_input = st.text_area(
    "Paste news article or headline here:",
    height=220,
    placeholder="Enter the news text here..."
)

col_meta1, col_meta2 = st.columns(2)
word_count = len(news_input.split()) if news_input.strip() else 0
char_count = len(news_input)

with col_meta1:
    st.caption(f"Word Count: {word_count}")
with col_meta2:
    st.caption(f"Character Count: {char_count}")

# Predict Button
if st.button("🔍 Verify Authenticity", use_container_width=True):
    if not news_input.strip():
        st.warning("Please enter some text first!")
    elif word_count < 5:
        st.warning("Please enter at least 5 words to allow accurate feature extraction.")
    else:
        with st.spinner("Analyzing text patterns and vocabulary..."):
            cleaned = clean_text(news_input)
            vec_input = tfidf.transform([cleaned])
            
            # Prediction
            prediction = model.predict(vec_input)[0]
            
            # Calculate confidence score
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(vec_input)[0]
                confidence = max(probs) * 100
            elif hasattr(model, "decision_function"):
                dist = model.decision_function(vec_input)[0]
                # Sigmoid transform for confidence
                prob_real = 1 / (1 + np.exp(-dist))
                confidence = prob_real if prediction == 1 else (1 - prob_real)
                confidence = float(np.clip(confidence * 100, 55.0, 99.5))
            else:
                confidence = 85.0

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if prediction == 1:
                st.success("### ✅ Classification: REAL NEWS")
                st.write("This article uses objective language and structured patterns typical of authentic reporting.")
            else:
                st.error("### 🚨 Classification: FAKE NEWS")
                st.write("This text matches patterns often associated with sensationalist, misleading, or fabricated articles.")

        with res_col2:
            st.metric("Model Confidence", f"{confidence:.1f}%")
            st.progress(int(confidence))