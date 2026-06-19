"""
Twitter Sentiment Analysis - Model Training Pipeline
This script preprocesses the generated synthetic tweets, extracts features using TF-IDF,
and trains multiple machine learning classifiers (Logistic Regression and Multinomial Naive Bayes).
It evaluates the models using classification metrics and serializes the models, vectorizer, and
evaluation metrics to disk for use in the Streamlit web application.
"""

import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Loading training dataset...")
df = pd.read_csv("data/tweets.csv")
print("Dataset size:", df.shape)

# Setup NLTK preprocessing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        
    STOPWORDS = set(stopwords.words("english"))
    # Keep some negative context terms that change sentiment
    STOPWORDS.discard("not")
    STOPWORDS.discard("no")
    STOPWORDS.discard("won")
    STOPWORDS.discard("but")
    
    lemmatizer = WordNetLemmatizer()
    USE_NLTK = True
    print("NLTK successfully configured.")
except Exception as e:
    print("NLTK configuration failed, using fallback preprocessing. Error:", e)
    FALLBACK_STOPWORDS = set('''
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's i i'd i'll i'm i've
    if in into is isn't it it's its itself let's me more most mustn't my myself
    nor of off on once only or other ought our ours ourselves out over
    own same shan't she she'd she'll she's should shouldn't so some such than
    that that's the their theirs them themselves then there there's these they
    they'd they'll they're they've this those through to too under until up
    very was wasn't we we'd we'll we're we've were weren't what what's when
    when's where where's which while who who's whom why why's with won't would
    wouldn't you you'd you'll you're you've your yours yourself yourselves
    '''.split())
    STOPWORDS = FALLBACK_STOPWORDS
    lemmatizer = None
    USE_NLTK = False

def clean_tweet(text):
    text = str(text)
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # 3. Remove Twitter handles (@username)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    # 4. Remove special characters and digits, replace punctuation with spaces
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    
    # 5. Tokenize, remove stopwords, and lemmatize
    tokens = text.split()
    if USE_NLTK:
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in STOPWORDS]
    else:
        tokens = [word for word in tokens if word not in STOPWORDS]
        
    return " ".join(tokens)

print("Preprocessing tweets...")
df["clean_text"] = df["text"].apply(clean_tweet)

print("Splitting train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["sentiment"],
    test_size=0.2, random_state=42, stratify=df["sentiment"]
)

print("Vectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training Logistic Regression Model...")
lr_model = LogisticRegression(max_iter=1000, C=1.0)
lr_model.fit(X_train_tfidf, y_train)

print("Training Multinomial Naive Bayes Model...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

print("Evaluating Models...")
for model_name, model in [("Logistic Regression", lr_model), ("Multinomial Naive Bayes", nb_model)]:
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    # Use macro average for multi-class classification evaluation
    prec = precision_score(y_test, preds, average='macro')
    rec = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')
    
    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

# Save models
print("\nSaving model files to disk...")
joblib.dump(lr_model, "lr_model.pkl")
joblib.dump(nb_model, "nb_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Save a file with metrics details to load directly in training UI
metrics = {
    "lr_acc": accuracy_score(y_test, lr_model.predict(X_test_tfidf)),
    "lr_prec": precision_score(y_test, lr_model.predict(X_test_tfidf), average='macro'),
    "lr_rec": recall_score(y_test, lr_model.predict(X_test_tfidf), average='macro'),
    "lr_f1": f1_score(y_test, lr_model.predict(X_test_tfidf), average='macro'),
    "nb_acc": accuracy_score(y_test, nb_model.predict(X_test_tfidf)),
    "nb_prec": precision_score(y_test, nb_model.predict(X_test_tfidf), average='macro'),
    "nb_rec": recall_score(y_test, nb_model.predict(X_test_tfidf), average='macro'),
    "nb_f1": f1_score(y_test, nb_model.predict(X_test_tfidf), average='macro'),
    "test_size": len(y_test),
    "train_size": len(y_train)
}
joblib.dump(metrics, "model_metrics.pkl")

print("All models successfully trained and serialized!")
