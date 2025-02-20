import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("cv_dataset.csv")

# Download stopwords if not available
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Preprocessing function
def preprocess_text(text):
    if isinstance(text, str):  # Ensure text is a string
        words = text.lower().split()
        return " ".join([word for word in words if word not in stop_words])
    return ""

# Apply preprocessing
df["cv_text"] = df["cv_text"].fillna("").apply(preprocess_text)

# Handle missing labels
df = df[df["Classification"].notna()]  # Remove NaN rows from classification column

# Map classification labels
y = df["Classification"].map({"Pass": 1, "Fail": 0})

# Ensure no NaN values in y
if y.isna().sum() > 0:
    print("Warning: NaN values detected in target labels, removing them.")
    df = df[y.notna()]  # Remove rows with NaN labels
    y = y[y.notna()]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cv_text"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Ensure model directory exists
model_dir = "models/classification"
os.makedirs(model_dir, exist_ok=True)

# Save the trained model and vectorizer
joblib.dump(model, os.path.join(model_dir, "naive_bayes.pkl"))
joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))

print("✅ Model and vectorizer saved successfully!")
