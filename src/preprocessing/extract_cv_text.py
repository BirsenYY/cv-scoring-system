import os
import pdfplumber
import pandas as pd

cv_dir = "data/generated_train_CVs"
df_labels = pd.read_csv(os.path.join(cv_dir, "cv_labels.csv"))

def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text
    except:
        return ""

df_labels["cv_text"] = df_labels["CV_Name"].apply(lambda x: extract_text_from_pdf(os.path.join(cv_dir, x)))
df_labels.to_csv("cv_dataset.csv", index=False)
print("✅ CV dataset created successfully!")

