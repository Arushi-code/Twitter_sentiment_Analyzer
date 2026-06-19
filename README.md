Twitter Data Sentiment Analyzer

Overview

Twitter Data Sentiment Analyzer is a Machine Learning and Natural Language Processing (NLP) project that classifies tweets into Positive, Negative, and Neutral sentiments.

The project uses TF-IDF feature extraction along with Logistic Regression and Multinomial Naive Bayes models to analyze sentiment and display results through an interactive Streamlit dashboard.

---

Features

- Sentiment classification of tweets
- Single tweet sentiment prediction
- Data preprocessing and cleaning
- TF-IDF feature extraction
- Logistic Regression Model
- Multinomial Naive Bayes Model
- Interactive Streamlit Dashboard
- Sentiment visualizations and statistics
- Model performance comparison
- Optional Twitter/X API integration

---

Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TextBlob
- VaderSentiment
- Tweepy
- Matplotlib
- Seaborn
- WordCloud
- Joblib

---

Project Structure

Twitter_Data_Sentiment_Analyzer/
│
├── app.py
├── train_models.py
├── generate_dataset.py
├── requirements.txt
├── README.md
│
├── data/
│   └── tweets.csv
│
├── screenshots/
│   ├── dashboard.png
│   ├── single_tweet_analyzer.png
│   ├── charts.png
│   └── model_metrics.png
│
├── lr_model.pkl
├── nb_model.pkl
├── tfidf_vectorizer.pkl
└── model_metrics.pkl

---

Dataset

The dataset contains:

- Tweet ID
- Username
- Tweet Text
- Sentiment Label
- Created Date
- Likes Count
- Retweets Count

Sentiment Classes:

- Positive
- Negative
- Neutral

---

Machine Learning Pipeline

Data Preprocessing

- Lowercase conversion
- URL removal
- Username removal
- Punctuation removal
- Number removal
- Stopword removal
- Lemmatization using NLTK

Feature Extraction

TF-IDF Vectorization

- Maximum Features: 2500
- N-Gram Range: (1,2)

Models Used

Logistic Regression

A powerful linear classification algorithm used for multi-class sentiment prediction.

Multinomial Naive Bayes

A probabilistic classifier commonly used in text classification tasks.

---

Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

---

Installation

Clone Repository

git clone https://github.com/Arushi-code/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME

Install Dependencies

pip install -r requirements.txt

Train Models

python train_models.py

Run Application

streamlit run app.py

---

Screenshots

Dashboard

"Dashboard" (screenshots/dashboard.png)

Single Tweet Analyzer

"Single Tweet Analyzer" (screenshots/single_tweet_analyzer.png)

Sentiment Visualizations

"Charts" (screenshots/charts.png)

Model Performance Metrics

"Model Metrics" (screenshots/model_metrics.png)

---

Results

The application successfully classifies tweets into:

- Positive
- Negative
- Neutral

and provides interactive visualizations and model evaluation metrics.

---

Future Improvements

- Real-time Twitter/X sentiment tracking
- Deep Learning Models (LSTM/BERT)
- Multi-language sentiment analysis
- Cloud Deployment
- Advanced Analytics Dashboard

---

Internship Project

Developed as part of an AI/ML Internship project focusing on:

- Natural Language Processing
- Sentiment Analysis
- Machine Learning
- Data Visualization
- Streamlit Application Development

---

Author

Aarushi Jha

B.Tech Student | AI & Machine Learning Enthusiast

GitHub: https://github.com/Arushi-code