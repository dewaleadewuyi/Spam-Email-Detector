import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# 1. Load the raw email dataset
df = pd.read_csv("spam.csv", encoding='latin-1')  # update name if needed
df = df.rename(columns={'v1': 'label', 'v2': 'text'})  # for popular spam dataset

print(df.head())
print(df.info())
# Keep only needed columns
df = df[['label', 'text']]

# Encode labels: ham = 0, spam = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 2. Preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = text.strip()
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# 3. Split data
X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_vec)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
print("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# 7. Save model and vectorizer
joblib.dump(model, "spamModel_raw.pkl")
joblib.dump(vectorizer, "vectorizer_raw.pkl")
print("\n✅ Model and vectorizer saved for real-world prediction!")
