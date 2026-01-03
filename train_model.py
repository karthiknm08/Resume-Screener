import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import re

# Example clean function
def cleanResume(txt):
    cleanText = re.sub(r'http\S+', ' ', txt)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

# Example dataset of resumes (replace with your own)
resumes = [
    "Experienced Python developer with ML skills",
    "HR professional with recruitment experience",
    "Front-end engineer with React and CSS experience"
]
categories = ["Developer", "HR", "Developer"]

# Clean resumes
resumes_cleaned = [cleanResume(r) for r in resumes]

# Vectorize
tfidf = TfidfVectorizer()
X_vect = tfidf.fit_transform(resumes_cleaned)

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(categories)

# Train model
svc_model = SVC(probability=True)
svc_model.fit(X_vect, y_enc)

# Save files
pickle.dump(svc_model, open("clf.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("Files saved: clf.pkl, tfidf.pkl, encoder.pkl")
