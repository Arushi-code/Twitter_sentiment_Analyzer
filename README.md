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

## ▶️ Run the App

streamlit run app.py

## 📈 Workflow

1. Input Tweet / Fetch Tweets  
2. Preprocessing (clean + normalize text)  
3. Feature Extraction (TF-IDF)  
4. Model Prediction  
5. Visualization in Streamlit Dashboard



## ⚠️ Limitations

- The project uses a simulated tweet stream by default  
- Real-time Twitter/X API access requires developer credentials  
- API usage is subject to rate limits and platform restrictions  

---

## 🔮 Future Improvements

- Integrate live Twitter/X API streaming  
- Implement deep learning models (BERT / LSTM)  
- Add multi-language sentiment detection  
- Deploy on Streamlit Cloud / Hugging Face Spaces  
- Improve trend analysis and forecasting capabilities  

---

## 👩‍💻 Author

**Aarushi Jha**  
B.Tech Student | AI & Machine Learning Enthusiast  

---

## 📜 License

This project is intended for educational and internship portfolio purposes only.
