import os
import joblib
import pdfplumber
import pandas as pd

# Load trained model and vectorizer
model = joblib.load("models/classification/naive_bayes.pkl")
vectorizer = joblib.load("models/classification/tfidf_vectorizer.pkl")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text
    except:
        return ""

def classify_cv(cv_path):
    """Classify a single CV."""
    cv_text = extract_text_from_pdf(cv_path)
    if not cv_text:
        return "❌ Could not extract text from CV"

    text_vectorized = vectorizer.transform([cv_text])
    prediction = model.predict(text_vectorized)[0]

    return "Pass" if prediction == 1 else "Fail"

# Load CV file names
cv_dir = "data/generated_test_CVs"
cv_files = [f for f in os.listdir(cv_dir) if f.endswith(".pdf")][:20]  # Limit to 10 CVs

# Classify multiple CVs
results = []
for cv_file in cv_files:
    cv_path = os.path.join(cv_dir, cv_file)
    classification = classify_cv(cv_path)
    results.append({"CV_Name": cv_file, "Predicted_Class": classification})
    print(f"📄 {cv_file} → {classification}")

# Save classification results
df_results = pd.DataFrame(results)
df_results.to_csv("data/classification_results.csv", index=False)
print("✅ Classification results saved to 'data/classification_results.csv'")
