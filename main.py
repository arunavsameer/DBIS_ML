import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to read emails from a directory
def read_emails_from_directory(directory):
    emails = []
    labels = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='latin-1') as file:
            content = file.read()
            emails.append(content)
            # Labels based on filename prefix
            if 'spm' in filename:
                labels.append(1)
            else:
                labels.append(0)
    return emails, labels

# Reading train and test data
train_dir = '/run/media/arunav/Data/programming/DBIS_ML/train_test_mails/train-mails'
test_dir = '/run/media/arunav/Data/programming/DBIS_ML/train_test_mails/test-mails'

train_emails, train_labels = read_emails_from_directory(train_dir)
test_emails, test_labels = read_emails_from_directory(test_dir)

# Vectorizing the emails using TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_emails)
X_test = vectorizer.transform(test_emails)

y_train = train_labels
y_test = test_labels

# Initializing classifiers
classifiers = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Training and evaluating classifiers
results = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

# Displaying the results
results_df = pd.DataFrame(results).T
print(results_df)
