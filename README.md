# Spam-Email-Detector
A Python-based spam email/SMS classifier using text cleaning, TF-IDF, and Naive Bayes. Includes training and prediction scripts.



# Spam Email/SMS Classifier

A simple Python application to detect spam using text cleaning, TF-IDF vectorization, and Multinomial Naive Bayes.

Features

* Preprocesses messages (remove URLs, numbers, punctuation).
* Converts text to numerical features with TF-IDF.
* Trains and evaluates a Naive Bayes classifier.
* Saves model and vectorizer for future predictions.
* Includes script to classify new emails/SMS as spam or ham.

How to Use

1. Clone the repo:

   ```bash
   git clone https://github.com/<your-username>/spam-classifier.git
   cd spam-classifier
   ```
2. Install dependencies:

   ```bash
   pip install pandas scikit-learn joblib
   ```
3. Train or predict:

   ```bash
   python train_model.py     # Train the model
   python predict_email.py   # Predict new messages
   ```
