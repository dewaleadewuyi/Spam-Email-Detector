import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('email.csv')


print(df.head())
print(df.info())

# Features = all columns except 'Prediction'
X = df.drop(columns=['Prediction', 'Email No.'])

# Labels = 'Prediction'
y = df['Prediction']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy Score: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

#Save the model
joblib.dump(model, 'spam_classifier_model.pkl')
print("\nModel saved as spam_classifier_model.pkl")

