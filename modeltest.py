import joblib

# Load model and vectorizer
model = joblib.load("spamModel_raw.pkl")
vectorizer = joblib.load("vectorizer_raw.pkl")

def predict_email(message):
    def clean_text(text):
        import re, string
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(f"[{string.punctuation}]", "", text)
        return text.strip()
    
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return "Spam" if pred == 1 else "Not Spam"


# Test the prediction function with multiple emails
emails = [
    "Congratulations! You've won a free gift card. Click here to claim now!",
    "Dear user, your account statement is attached.",
    "Limited time offer! Buy now and save 50%.",
    "Hi John, just checking in about our meeting tomorrow.",
    "URGENT: Your account has been compromised. Reset your password immediately.",
    "Lunch at 1pm? Let me know if that works for you.",
    "You have been selected for a cash prize. Reply with your bank details.",
    "Project update: all tasks are on track for completion.",
    "Win a brand new iPhone by entering our survey!",
    "Please review the attached invoice and confirm receipt."
]

for i, email in enumerate(emails, 1):
    print(f"Email {i}: {predict_email(email)}\n  Text: {email}\n")
