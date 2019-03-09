# -*- coding: utf-8 -*-
 
import os
import re
import ssl

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Todo: Not use absolute paths!
os.chdir("/Users/ebalboa/Documents/AI-SATURDAYS/20190309_nlp/2016-campaign/")

# movie_data = load_files(r"./../texts")

speech_data = load_files(r"texts/")
# 'target_names': ['Clinton', 'Trump']

X, y = speech_data.data, speech_data.target
# speech_data.target gives an array of zeros and ones.
print(y.__len__())

documents = []

try:
    """
    Try is used for mac users that face with a ssl certificate issue, to let them download from nltk library
    """
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()


# nltk.download('stopwords')
# nltk.download('wordnet')

stemmer = WordNetLemmatizer()

print("Stopwords and wordnet downloaded successful")

for sen in range(0, len(X)):
    
    # Remove all the special characters
    document: str = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Removing unwanted characters
    document = re.sub(r'x9[0-9]', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

tfidfconverter = TfidfVectorizer(max_features=100, min_df=1, max_df=0.9, stop_words=stopwords.words('english'))  
tfidfconverter.fit(documents)
X = tfidfconverter.transform(documents).toarray()

#X = tfidfconverter.fit_transform(documents).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test) 


print("Confusion Matrix: ")
print(confusion_matrix(y_test,y_pred))

print("Classification Report:c")
print(classification_report(y_test,y_pred))

print("Accuracy Score achieved: " + str(accuracy_score(y_test, y_pred)))
