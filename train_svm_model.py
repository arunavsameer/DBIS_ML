import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Function to read emails from a directory
def read_emails_from_directory(directory):
    emails = []
    labels = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='latin-1') as file:
            content = file.read()
            emails.append(content)
            if 'spm' in filename:
                labels.append(1)
            else:
                labels.append(0)
    return emails, labels

# Directories
train_dir = '/run/media/arunav/Data/programming/DBIS_ML/train_test_mails/train-mails'

# Reading training data
train_emails, train_labels = read_emails_from_directory(train_dir)

# Vectorizing the emails using TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_emails)

# Training the SVM classifier
svm_clf = SVC(probability=True)
svm_clf.fit(X_train, train_labels)

# Saving the model and vectorizer
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm_clf, model_file)
with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)
