# 🐦 Twitter Sentiment Analyzer

## 📌 Overview

Twitter Sentiment Analyzer is a Machine Learning + NLP web application that classifies tweets into:

- 😊 Positive  
- 😐 Neutral  
- 😡 Negative  

It combines traditional ML models and lexicon-based methods with an interactive Streamlit dashboard for real-time sentiment insights.

---

## 🚀 Features

### 📊 Real-Time Dashboard
- Analyze tweets using hashtags or keywords  
- Live sentiment distribution charts  
- Trend analysis over time  
- Word clouds for positive & negative words  

### ✍️ Single Tweet Analysis
- Input custom tweet text  
- Compare multiple models:
  - VADER  
  - TextBlob  
  - Logistic Regression  
  - Multinomial Naive Bayes  

### 🧠 NLP Pipeline
- Text cleaning (URLs, mentions, punctuation)  
- Lowercasing  
- Stopword removal  
- Lemmatization  
- TF-IDF vectorization  

### 🤖 Machine Learning Models
- Logistic Regression  
- Multinomial Naive Bayes  
- Model evaluation metrics  
- Retraining support  

---

## 🧰 Tech Stack

- Python  
- Streamlit  
- Scikit-learn  
- Pandas & NumPy  
- Matplotlib & Seaborn  
- NLTK  
- TextBlob  
- VADER Sentiment  

---

## 📂 Project Structure

Twitter_sentiment_Analyzer/ │ ├── app.py ├── train_models.py ├── generate_dataset.py ├── requirements.txt ├── README.md │ ├── data/ │ └── tweets.csv │ ├── assets/ ├── screenshots/ │ ├── lr_model.pkl ├── nb_model.pkl ├── tfidf_vectorizer.pkl └── model_metrics.pkl

---

## ⚙️ Installation

git clone https://github.com/Arushi-code/Twitter_sentiment_Analyzer.git
cd Twitter_sentiment_Analyzer

python -m venv myenv
myenv\Scripts\activate   # Windows

pip install -r requirements.txt

---

▶️ Run the App

---

streamlit run app.py

---

📈 Workflow

---

Input Tweet / Fetch Tweets
Preprocessing (clean + normalize text)
Feature Extraction (TF-IDF)
Model Prediction
Visualization in Streamlit

---

⚠️ Limitations

---

Uses simulated tweet stream by default
Real-time Twitter/X API requires developer access
API rate limits may apply

---

🔮 Future Improvements

---

Live Twitter API integration
BERT / LSTM deep learning models
Multi-language sentiment detection
Deployment on Streamlit Cloud
Advanced trend forecasting

---

👩‍💻 Author

---

Aarushi Jha
B.Tech Student | AI & Machine Learning Enthusiast


📜 License

---

This project is for educational and internship portfolio use.
