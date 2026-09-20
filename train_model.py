import re
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Download Stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

print("=" * 60)
print("1. Loading Datasets...")
print("=" * 60)

if not os.path.exists('True.csv') or not os.path.exists('Fake.csv'):
    raise FileNotFoundError("True.csv ya Fake.csv folder me nahi mili!")

df_true = pd.read_csv('True.csv')
df_fake = pd.read_csv('Fake.csv')

df_true['label'] = 1  # Real
df_fake['label'] = 0  # Fake

df = pd.concat([df_true, df_fake], axis=0).reset_index(drop=True)
df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')

# Fast training sample (10,000 articles)
sample_size = min(10000, len(df))
df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
print(f"Total processed samples: {len(df)}")

print("\n2. Cleaning and Preprocessing text...")
df['clean_content'] = df['content'].apply(clean_text)

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_content'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

print("\n3. Vectorizing with TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000, max_df=0.7, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 4. Multi-Model Evaluation
print("\n4. Training and Evaluating Models...")
models = {
    "Passive Aggressive": PassiveAggressiveClassifier(max_iter=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Multinomial Naive Bayes": MultinomialNB()
}

results = {}
trained_models = {}

for name, clf in models.items():
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    trained_models[name] = clf
    print(f"-> {name} Accuracy: {acc * 100:.2f}%")

# Select best model
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]
print(f"\nBest Model Selected: {best_model_name} ({results[best_model_name] * 100:.2f}%)")

# Generate and Save Confusion Matrix Graph for Project Report
y_best_pred = best_model.predict(X_test_vec)
cm = confusion_matrix(y_test, y_best_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title(f'Confusion Matrix ({best_model_name})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
print("Saved confusion matrix image as 'confusion_matrix.png'")

# Model Comparison Bar Chart
plt.figure(figsize=(7, 4))
bars = plt.bar(results.keys(), [v * 100 for v in results.values()], color=['#2b5c8f', '#4682b4', '#87ceeb'])
plt.ylabel('Accuracy (%)')
plt.ylim([80, 100])
plt.title('Algorithm Comparison')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.close()
print("Saved model comparison image as 'model_comparison.png'")

# 5. Export Best Model & Vectorizer
joblib.dump(best_model, 'model.pkl')
joblib.dump(tfidf, 'vectorizer.pkl')
print("\nSuccess: 'model.pkl' and 'vectorizer.pkl' exported successfully!")
print("=" * 60)