# CV Scoring System
A machine learning-based CV scoring system that automates resume classification and evaluation, starting with Naive Bayes and evolving with advanced models and features.

## Automated CV Classification System 📄✨
Naive Bayes, TF-IDF vectorization, and NLP preprocessing were used to evaluate resumes efficiently.

## Features
✅ Extract text from PDFs using pdfplumber
✅ Preprocess text (tokenization, stopwords removal) with nltk
✅ Vectorize text using TfidfVectorizer from scikit-learn
✅ Train & evaluate a Naive Bayes model for CV classification
✅ Save & load models with joblib

 ## Installation
1. Prerequisites
Python 3.11
Conda (for environment management)
Git (optional, for cloning the repo)
2. Clone the Repository

git clone https://github.com/your-username/cv-scoring-system.git
cd cv-scoring-system

3. Create & Activate the Conda Environment

conda create --name cv_env python=3.11
conda activate cv_env

4. Install Dependencies

pip install -r requirements.txt

## Notes
The project uses TF-IDF vectorization and Naive Bayes classification.
Model is saved using joblib and can be reloaded for future use.
Ensure you have an .env file to store API keys securely (if needed).

## License
This project is licensed under the MIT License.




