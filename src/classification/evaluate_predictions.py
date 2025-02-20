import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load true labels and predicted labels
df_true = pd.read_csv("data/generated_test_CVs/cv_labels.csv")  # True labels
df_pred = pd.read_csv("data/classification_results.csv")  # Model predictions

# Merge both files based on CV_Name
df_merged = df_true.merge(df_pred, on="CV_Name")

# Convert labels to numerical values
df_merged["True_Class"] = df_merged["Classification"].map({"Pass": 1, "Fail": 0})
df_merged["Predicted_Class"] = df_merged["Predicted_Class"].map({"Pass": 1, "Fail": 0})

# Calculate accuracy
accuracy = accuracy_score(df_merged["True_Class"], df_merged["Predicted_Class"])
accuracy_str = f"✅ Model Accuracy on test CVs: {accuracy * 100:.2f}%\n"

# Generate classification report
classification_rep = classification_report(df_merged["True_Class"], df_merged["Predicted_Class"])

# Print results to console
print(accuracy_str)
print("\n📊 Classification Report:")
print(classification_rep)

# Write results to a text file
results_file = "classification_results.txt"
with open(results_file, "w") as file:
    file.write(accuracy_str + "\n")
    file.write("📊 Classification Report:\n")
    file.write(classification_rep + "\n")

print(f"📄 Results saved to {results_file} ✅")
