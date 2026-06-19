import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import string
import os
import time
import random
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import tweepy
from wordcloud import WordCloud
import subprocess
import sys

# Set page config for a premium layout
st.set_page_config(
    page_title="🐦 Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS (Glassmorphism & Harmonious Gradients)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Glassmorphism containers */
.card {
    background: rgba(255, 255, 255, 0.75);
    border-radius: 16px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.04);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.dark-card {
    background: rgba(15, 23, 42, 0.04);
    border-radius: 12px;
    padding: 1.25rem;
    border: 1px solid rgba(0, 0, 0, 0.05);
    margin-bottom: 1.25rem;
}

/* Header style */
.main-title {
    font-size: 2.85rem;
    font-weight: 800;
    background: linear-gradient(135deg, #1DA1F2 0%, #0C85D0 30%, #4f46e5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.1rem;
    letter-spacing: -0.03em;
}

.sub-title {
    font-size: 1.15rem;
    color: #4B5563;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Prediction boxes */
.result-container {
    border-radius: 12px;
    padding: 1.25rem;
    font-size: 1.35rem;
    font-weight: 700;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.02);
    margin-bottom: 1rem;
}

.result-positive {
    background-color: #ECFDF5;
    color: #059669;
    border: 1.5px solid #A7F3D0;
}

.result-negative {
    background-color: #FEF2F2;
    color: #DC2626;
    border: 1.5px solid #FCA5A5;
}

.result-neutral {
    background-color: #F9FAFB;
    color: #4B5563;
    border: 1.5px solid #E5E7EB;
}

/* Word highlight tags */
.word-highlight {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    margin: 0.15rem 0.2rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s ease;
}

.word-positive {
    background-color: rgba(16, 185, 129, 0.12);
    color: #059669;
    border-bottom: 2.5px solid #10B981;
}

.word-negative {
    background-color: rgba(239, 68, 68, 0.12);
    color: #DC2626;
    border-bottom: 2.5px solid #EF4444;
}

.word-neutral {
    color: #4B5563;
    border-bottom: 1.5px solid transparent;
}

/* Stat badges */
.stat-badge {
    background-color: #F3F4F6;
    padding: 0.4rem 0.9rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #4B5563;
    display: inline-block;
    margin-right: 0.5rem;
    border: 1px solid #E5E7EB;
}

/* Styled tags for lists */
.badge {
    padding: 0.25em 0.6em;
    font-size: 75%;
    font-weight: 700;
    line-height: 1;
    text-align: center;
    white-space: nowrap;
    vertical-align: baseline;
    border-radius: 0.375rem;
    display: inline-block;
    margin-right: 5px;
}
.badge-pos { background-color: #D1FAE5; color: #065F46; }
.badge-neg { background-color: #FEE2E2; color: #991B1B; }
.badge-neu { background-color: #F3F4F6; color: #374151; }

</style>
""", unsafe_allow_html=True)

# Define resources loading (ML models, Lexicon)
@st.cache_resource
def get_vader_analyzer():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_ml_resources():
    try:
        lr_model = joblib.load("lr_model.pkl")
        nb_model = joblib.load("nb_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        metrics = joblib.load("model_metrics.pkl")
        return lr_model, nb_model, vectorizer, metrics
    except Exception:
        return None, None, None, None

# Preprocessing configs
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
    STOPWORDS.discard("not")
    STOPWORDS.discard("no")
    STOPWORDS.discard("won")
    STOPWORDS.discard("but")
    lemmatizer = WordNetLemmatizer()
    USE_NLTK = True
except Exception:
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

def clean_tweet_text(text):
    text = str(text)
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove handles
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    # Punctuation and numbers
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    
    # Tokens, stopwords, lemmatization
    tokens = text.split()
    if USE_NLTK:
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in STOPWORDS]
    else:
        tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

# Load resources
vader_analyzer = get_vader_analyzer()
lr_model, nb_model, vectorizer, ml_metrics = load_ml_resources()

# Predefined Simulated Tweet Bank for Keyword Searches
simulated_tweet_templates = {
    "positive": [
        "Absolutely love {keyword}! It's an absolute game changer. 🚀 #innovation",
        "The latest update on {keyword} is amazing. Performance is so smooth and clean!",
        "Stunning tech. Genuinely impressed by how they built {keyword}. Kudos team! 🙌",
        "Really happy with the results of {keyword}. High quality product, 10/10.",
        "Had a great experience using {keyword} today. Highly recommended! #tech",
        "This is hands down the best implementation of {keyword} I have ever seen. 😍",
        "Congrats on the release. It's working flawlessly. #excited",
        "I'm super excited for the future of {keyword}. Big things ahead!",
        "Using {keyword} makes my workflow so much faster. Grateful for this. 🌟",
        "Wow, incredible support and updates on {keyword}. A perfect developer experience."
    ],
    "negative": [
        "Terrible experience with {keyword}. It crashed my system twice. 😡 #fail",
        "The new update for {keyword} is awful. It is laggy and full of bugs.",
        "Way overpriced for what it offers. Save your money and avoid {keyword}.",
        "Really frustrated with the downtime of {keyword} today. Lost hours of work. 😭",
        "This version of {keyword} is a major step backwards. hard to navigate.",
        "Extremely disappointed with {keyword}. The customer support ignores queries.",
        "I regret starting to use {keyword}. It's fragile and cheap.",
        "Why is {keyword} so slow today? Getting endless load times. #annoyed",
        "Another security warning regarding {keyword}? This is totally unacceptable.",
        "Very bad quality. Tried to integrate {keyword} but it keeps throwing exceptions."
    ],
    "neutral": [
        "Just read an interesting blog post comparing {keyword} with alternatives.",
        "Does anyone have the official documentation link for {keyword}?",
        "Currently testing the features of {keyword} for our team project.",
        "Rescheduled the meeting regarding the integration of {keyword}.",
        "A standard review of the pros and cons of {keyword} is out now.",
        "We are analyzing the market trends for {keyword} this quarter.",
        "Checking if this library supports the latest version of {keyword}.",
        "The package for {keyword} arrived today. Standard design, intact.",
        "Currently reading a research paper on the development of {keyword}.",
        "Please send your feedback on {keyword} by the end of the day. Thanks."
    ]
}

def generate_simulated_stream(keyword, num_tweets):
    # Determine general sentiment skew based on keyword to make it feel realistic
    kw_lower = keyword.lower()
    if any(neg in kw_lower for neg in ["bug", "crash", "bad", "fail", "slow", "down", "error"]):
        weights = [0.15, 0.70, 0.15] # Positive, Negative, Neutral
    elif any(pos in kw_lower for pos in ["love", "great", "best", "awesome", "good", "win", "excited"]):
        weights = [0.70, 0.10, 0.20]
    elif any(neu in kw_lower for neu in ["update", "version", "tutorial", "meeting", "check", "info"]):
        weights = [0.20, 0.15, 0.65]
    else:
        weights = [0.45, 0.25, 0.30] # default mixed
        
    tweets = []
    usernames = ["tech_enthusiast", "cryptoking", "code_geek", "design_guru", "daily_news", "reviewer_pro", "investor_daily", "dev_master", "startup_coder", "web_architect"]
    
    # Generate tweets
    for i in range(num_tweets):
        sentiment_type = np.random.choice(["positive", "negative", "neutral"], p=weights)
        template = random.choice(simulated_tweet_templates[sentiment_type])
        text = template.replace("{keyword}", keyword)
        
        # Add random handles/hashtags
        if random.random() < 0.3:
            text = f"@{random.choice(usernames)} " + text
        if random.random() < 0.2:
            text = text + f" #{keyword.replace(' ', '').replace('#', '')}"
            
        likes = int(np.random.exponential(scale=120))
        retweets = int(likes * np.random.uniform(0.1, 0.35))
        
        # Staggered dates in the last 24 hours
        time_diff = timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
        created_at = datetime.now() - time_diff
        
        tweets.append({
            "tweet_id": str(random.randint(10**17, 10**18 - 1)),
            "username": random.choice(usernames),
            "text": text,
            "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "likes": likes,
            "retweets": retweets
        })
        
    # Sort by date descending
    tweets = sorted(tweets, key=lambda x: x["created_at"], reverse=True)
    return pd.DataFrame(tweets)

# Setup Sidebar Configurations
st.sidebar.markdown('<div style="text-align: center;"><h2 style="color:#1DA1F2; margin-bottom: 0;">🛠️ X Config</h2></div>', unsafe_allow_html=True)
st.sidebar.caption("Define your data acquisition preferences below:")

stream_mode = st.sidebar.selectbox(
    "Data Source Mode",
    ["Simulated Stream (Instant)", "Live Twitter API (Tweepy)"]
)

# Render Tweepy Credentials config if selected
tweepy_api = None
if stream_mode == "Live Twitter API (Tweepy)":
    st.sidebar.warning("API credentials are required for live streaming.")
    with st.sidebar.expander("🔑 Twitter Developer Credentials", expanded=True):
        bearer_token = st.text_input("Bearer Token", type="password")
        api_key = st.text_input("API Key", type="password")
        api_secret = st.text_input("API Secret Key", type="password")
        access_token = st.text_input("Access Token", type="password")
        access_secret = st.text_input("Access Token Secret", type="password")
        
        if bearer_token and api_key and api_secret and access_token and access_secret:
            try:
                # Initialize Tweepy Client
                client = tweepy.Client(
                    bearer_token=bearer_token,
                    consumer_key=api_key,
                    consumer_secret=api_secret,
                    access_token=access_token,
                    access_token_secret=access_secret
                )
                # Quick test connection
                st.sidebar.success("Tweepy API ready!")
                tweepy_api = client
            except Exception as e:
                st.sidebar.error(f"Failed to connect: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏷️ Search Parameters")
search_keyword = st.sidebar.text_input("Hashtag or Keyword", value="#ArtificialIntelligence")
max_tweets = st.sidebar.slider("Number of Tweets", min_value=10, max_value=150, value=50, step=10)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Active Classifier")
classifier_choice = st.sidebar.selectbox(
    "Select Model for Analysis",
    ["VADER (Rule-Based)", "TextBlob (Polarity-Based)", "Custom ML (Logistic Regression)", "Custom ML (Naive Bayes)"]
)

if classifier_choice == "Custom ML (Logistic Regression)" and lr_model is None:
    st.sidebar.error("⚠️ Custom ML model not trained yet. Go to the 'ML Training Center' tab first!")
elif classifier_choice == "Custom ML (Naive Bayes)" and nb_model is None:
    st.sidebar.error("⚠️ Custom ML model not trained yet. Go to the 'ML Training Center' tab first!")

# Title and Subtitle
st.markdown('<div class="main-title">🐦 Twitter Sentiment Analyzer Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A Premium Real-Time NLP Pipeline and Machine Learning Analysis System</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_dash, tab_single, tab_nlp, tab_ml = st.tabs([
    "📊 Real-time Dashboard", 
    "🔍 Single Tweet Analyzer", 
    "⚙️ Preprocessing Pipeline", 
    "🧠 ML Model Training"
])

# Utility function to analyze a list of tweets
def analyze_sentiment_batch(texts, method):
    results = []
    polarities = []
    
    for text in texts:
        cleaned = clean_tweet_text(text)
        
        if method == "VADER (Rule-Based)":
            scores = vader_analyzer.polarity_scores(text) # VADER runs on raw text (preserves punctuation/emoji nuance)
            compound = scores['compound']
            polarities.append(compound)
            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"
                
        elif method == "TextBlob (Polarity-Based)":
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            polarities.append(polarity)
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"
                
        elif method == "Custom ML (Logistic Regression)":
            if lr_model is not None and vectorizer is not None:
                vec = vectorizer.transform([cleaned])
                label = lr_model.predict(vec)[0]
                # Map probabilities to a continuous polarity metric [-1 to 1] for trend charting
                probs = lr_model.predict_proba(vec)[0]
                # Classes: ['negative', 'neutral', 'positive']
                compound = probs[2] - probs[0]
                polarities.append(compound)
            else:
                label = "neutral"
                polarities.append(0.0)
        elif method == "Custom ML (Naive Bayes)":
            if nb_model is not None and vectorizer is not None:
                vec = vectorizer.transform([cleaned])
                label = nb_model.predict(vec)[0]
                # Map probabilities to a continuous polarity metric [-1 to 1] for trend charting
                probs = nb_model.predict_proba(vec)[0]
                # Classes: ['negative', 'neutral', 'positive']
                compound = probs[2] - probs[0]
                polarities.append(compound)
            else:
                label = "neutral"
                polarities.append(0.0)
        else:
            label = "neutral"
            polarities.append(0.0)
                
        results.append(label)
        
    return results, polarities

# --- TAB 1: Real-time Dashboard ---
with tab_dash:
    col_ctrl, col_space = st.columns([1, 4])
    with col_ctrl:
        fetch_clicked = st.button("🚀 Analyze X Stream", type="primary", use_container_width=True)
        
    if "df_results" not in st.session_state or fetch_clicked:
        if fetch_clicked or "df_results" not in st.session_state:
            with st.spinner("Fetching and analyzing data stream..."):
                time.sleep(1.2) # loading effect
                
                if stream_mode == "Live Twitter API (Tweepy)":
                    if tweepy_api is None:
                        st.error("Please configure your Tweepy API credentials in the sidebar first!")
                        st.stop()
                    try:
                        # Fetch live tweets using Twitter API Client v2
                        # search_recent_tweets returns fields: created_at, public_metrics
                        query = f"{search_keyword} -is:retweet lang:en"
                        response = tweepy_api.search_recent_tweets(
                            query=query,
                            max_results=min(max_tweets, 100), # v2 limits recent search to 10-100 per call
                            tweet_fields=["created_at", "public_metrics"],
                            expansions=["author_id"]
                        )
                        
                        if response.data:
                            # Map tweets data
                            tweets_list = []
                            for tweet in response.data:
                                metrics = tweet.public_metrics
                                tweets_list.append({
                                    "tweet_id": str(tweet.id),
                                    "username": f"user_{tweet.author_id[:8]}" if tweet.author_id else "anonymous",
                                    "text": tweet.text,
                                    "created_at": tweet.created_at.strftime("%Y-%m-%d %H:%M:%S") if tweet.created_at else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "likes": metrics.get("like_count", 0),
                                    "retweets": metrics.get("retweet_count", 0)
                                })
                            df_stream = pd.DataFrame(tweets_list)
                        else:
                            st.warning("No recent tweets found for that query. Reverting to Simulated Stream.")
                            df_stream = generate_simulated_stream(search_keyword, max_tweets)
                    except Exception as e:
                        st.error(f"Tweepy Fetch Error: {str(e)}. Falling back to Simulated Stream.")
                        df_stream = generate_simulated_stream(search_keyword, max_tweets)
                else:
                    # Simulated Stream
                    df_stream = generate_simulated_stream(search_keyword, max_tweets)
                    
                # Run sentiment pipeline
                preds, pols = analyze_sentiment_batch(df_stream["text"], classifier_choice)
                df_stream["predicted_sentiment"] = preds
                df_stream["polarity_score"] = pols
                st.session_state.df_results = df_stream
                st.session_state.last_keyword = search_keyword
                
    df_results = st.session_state.df_results
    
    st.markdown(f"### 📈 Analysis Results for **{st.session_state.last_keyword}**")
    
    # Grid of Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    total_count = len(df_results)
    pos_count = len(df_results[df_results["predicted_sentiment"] == "positive"])
    neg_count = len(df_results[df_results["predicted_sentiment"] == "negative"])
    neu_count = len(df_results[df_results["predicted_sentiment"] == "neutral"])
    
    avg_pol = df_results["polarity_score"].mean()
    
    col1.metric("Tweets Analyzed", f"{total_count}")
    col2.metric("Positive Sentiment", f"{(pos_count/total_count)*100:.1f}%", delta=f"{pos_count} tweets", delta_color="normal")
    col3.metric("Negative Sentiment", f"{(neg_count/total_count)*100:.1f}%", delta=f"-{neg_count} tweets", delta_color="inverse")
    col4.metric("Neutral Sentiment", f"{(neu_count/total_count)*100:.1f}%", f"{neu_count} tweets")
    col5.metric("Avg Polarity Score", f"{avg_pol:.3f}", 
                delta="Positive Skew" if avg_pol > 0.05 else ("Negative Skew" if avg_pol < -0.05 else "Neutral"))
    
    # Layout with Charts
    c_col1, c_col2 = st.columns([1, 1])
    
    with c_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🍩 Sentiment Distribution")
        
        # Donut Chart with modern colors
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [pos_count, neg_count, neu_count]
        colors = ['#10B981', '#EF4444', '#9CA3AF']
        
        # Only plot categories with counts > 0
        active_indices = [i for i, size in enumerate(sizes) if size > 0]
        active_labels = [labels[i] for i in active_indices]
        active_sizes = [sizes[i] for i in active_indices]
        active_colors = [colors[i] for i in active_indices]
        
        ax.pie(active_sizes, labels=active_labels, autopct='%1.1f%%', startangle=90, 
               colors=active_colors, pctdistance=0.75, textprops={'fontweight':'bold', 'fontsize':11})
        # Create a circle in the center to make it a donut
        centre_circle = plt.Circle((0,0),0.55,fc='white')
        fig.gca().add_artist(centre_circle)
        ax.axis('equal')  
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ⏳ Sentiment Polarity Trend (Last 24 Hours)")
        
        # Convert created_at to datetime for plotting
        df_sorted = df_results.copy()
        df_sorted["datetime"] = pd.to_datetime(df_sorted["created_at"])
        df_sorted = df_sorted.sort_values(by="datetime")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Rolling average trend line
        ax.plot(df_sorted["datetime"], df_sorted["polarity_score"], 'o-', color='#3B82F6', alpha=0.3, label="Tweet Polarity")
        if len(df_sorted) >= 5:
            df_sorted["rolling"] = df_sorted["polarity_score"].rolling(window=max(5, len(df_sorted)//5), min_periods=1).mean()
            ax.plot(df_sorted["datetime"], df_sorted["rolling"], '-', color='#1E40AF', linewidth=2.5, label="Rolling Trend")
            
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Polarity Score (Negative ➔ Positive)", fontsize=10)
        ax.set_xlabel("Time", fontsize=10)
        plt.xticks(rotation=30)
        ax.legend(loc="upper left")
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Word frequency plot
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🔠 Top Sentiment Keywords")
    
    # Collect words
    pos_words = []
    neg_words = []
    for idx, row in df_results.iterrows():
        cleaned_w = clean_tweet_text(row["text"]).split()
        if row["predicted_sentiment"] == "positive":
            pos_words.extend(cleaned_w)
        elif row["predicted_sentiment"] == "negative":
            neg_words.extend(cleaned_w)
            
    # Filter out empty or query keyword itself to make cloud interesting
    query_terms = set(search_keyword.lower().replace('#', '').split())
    
    # Toggle between Bar Charts and Word Clouds
    viz_style = st.radio("Choose Visualization Style", ["Bar Charts", "Word Clouds"], horizontal=True)
    
    col_w1, col_w2 = st.columns(2)
    
    pos_words_filtered = [w for w in pos_words if w not in query_terms and len(w) > 2]
    neg_words_filtered = [w for w in neg_words if w not in query_terms and len(w) > 2]
    
    if viz_style == "Bar Charts":
        with col_w1:
            st.markdown("##### 🟢 Top Positive Terms")
            if pos_words_filtered:
                pos_counts = pd.Series(pos_words_filtered).value_counts().head(12)
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.barplot(x=pos_counts.values, y=pos_counts.index, color='#10B981', ax=ax)
                ax.set_xlabel("Count")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.caption("No positive terms found in batch.")
                
        with col_w2:
            st.markdown("##### 🔴 Top Negative Terms")
            if neg_words_filtered:
                neg_counts = pd.Series(neg_words_filtered).value_counts().head(12)
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.barplot(x=neg_counts.values, y=neg_counts.index, color='#EF4444', ax=ax)
                ax.set_xlabel("Count")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.caption("No negative terms found in batch.")
    else:
        # Word Clouds
        with col_w1:
            st.markdown("##### 🟢 Positive Word Cloud")
            if pos_words_filtered:
                pos_text = " ".join(pos_words_filtered)
                wc_pos = WordCloud(width=600, height=300, background_color='white', colormap='summer', max_words=50).generate(pos_text)
                fig_wc, ax_wc = plt.subplots(figsize=(6, 3))
                ax_wc.imshow(wc_pos, interpolation='bilinear')
                ax_wc.axis('off')
                plt.tight_layout()
                st.pyplot(fig_wc)
            else:
                st.caption("No positive terms found to generate Word Cloud.")
                
        with col_w2:
            st.markdown("##### 🔴 Negative Word Cloud")
            if neg_words_filtered:
                neg_text = " ".join(neg_words_filtered)
                wc_neg = WordCloud(width=600, height=300, background_color='white', colormap='autumn', max_words=50).generate(neg_text)
                fig_wc, ax_wc = plt.subplots(figsize=(6, 3))
                ax_wc.imshow(wc_neg, interpolation='bilinear')
                ax_wc.axis('off')
                plt.tight_layout()
                st.pyplot(fig_wc)
            else:
                st.caption("No negative terms found to generate Word Cloud.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Raw Tweets Table
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📜 Stream Feed Data Table")
    
    # Filter controls
    sent_filter = st.multiselect("Filter Sentiment Feed", ["positive", "negative", "neutral"], default=["positive", "negative", "neutral"])
    df_filtered = df_results[df_results["predicted_sentiment"].isin(sent_filter)]
    
    # Display styled list
    for idx, row in df_filtered.iterrows():
        badge_style = "badge-pos" if row["predicted_sentiment"] == "positive" else ("badge-neg" if row["predicted_sentiment"] == "negative" else "badge-neu")
        badge_label = row["predicted_sentiment"].upper()
        
        st.markdown(
            f"""
            <div style="border-bottom: 1px solid #E5E7EB; padding: 0.8rem 0; display: flex; flex-direction: column; gap: 4px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-weight: 700; color: #1DA1F2;">@{row['username']}</span>
                    <span style="font-size: 0.8rem; color: #9CA3AF;">{row['created_at']}</span>
                    <span class="badge {badge_style}">{badge_label}</span>
                    <span style="font-size: 0.8rem; color: #6B7280; margin-left: auto;">❤️ {row['likes']} | 🔁 {row['retweets']}</span>
                </div>
                <div style="font-size: 1rem; color: #1F2937;">{row['text']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)


# --- TAB 2: Single Tweet Analyzer ---
with tab_single:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Test Custom Tweet Sentiment")
    st.write("Type or paste any tweet text to run through the text processing pipeline and evaluate sentiment scores:")
    
    custom_tweet = st.text_area("Tweet Text", placeholder="Write your tweet here...", height=100)
    
    col_ab1, col_ab2 = st.columns([1, 6])
    with col_ab1:
        analyze_clicked = st.button("Evaluate Sentiment", type="primary", use_container_width=True)
    with col_ab2:
        if st.button("Load Positive Preset", use_container_width=True):
            custom_tweet = "The customer service for ChatGPT is brilliant! Answered my queries in under 5 minutes and solved the issue immediately. Loving it! 🌟"
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)
    
    if analyze_clicked or custom_tweet.strip() != "":
        if custom_tweet.strip() == "":
            st.warning("Please type some text to analyze.")
        else:
            cleaned_text = clean_tweet_text(custom_tweet)
            
            # Predict with all models
            # 1. VADER
            v_scores = vader_analyzer.polarity_scores(custom_tweet)
            v_compound = v_scores['compound']
            v_label = "positive" if v_compound >= 0.05 else ("negative" if v_compound <= -0.05 else "neutral")
            
            # 2. TextBlob
            tb_polarity = TextBlob(custom_tweet).sentiment.polarity
            tb_label = "positive" if tb_polarity > 0.1 else ("negative" if tb_polarity < -0.1 else "neutral")
            
            # 3. Custom Logistic Regression ML
            if lr_model is not None and vectorizer is not None:
                vec = vectorizer.transform([cleaned_text])
                ml_label = lr_model.predict(vec)[0]
                ml_probs = lr_model.predict_proba(vec)[0] # negative, neutral, positive
                ml_conf = ml_probs[2] if ml_label == "positive" else (ml_probs[0] if ml_label == "negative" else ml_probs[1])
            else:
                ml_label = "ML model not trained"
                ml_conf = 0.0

            # 4. Custom Naive Bayes ML
            if nb_model is not None and vectorizer is not None:
                vec = vectorizer.transform([cleaned_text])
                nb_label = nb_model.predict(vec)[0]
                nb_probs = nb_model.predict_proba(vec)[0] # negative, neutral, positive
                nb_conf = nb_probs[2] if nb_label == "positive" else (nb_probs[0] if nb_label == "negative" else nb_probs[1])
            else:
                nb_label = "ML model not trained"
                nb_conf = 0.0
                
            # Render side-by-side predictions
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🏷️ Classifier Predictions")
            
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            
            with col_p1:
                st.markdown('<div class="dark-card" style="text-align: center;">', unsafe_allow_html=True)
                st.markdown("##### VADER (Rule-Based)")
                c_style = "result-positive" if v_label == "positive" else ("result-negative" if v_label == "negative" else "result-neutral")
                st.markdown(f'<div class="result-container {c_style}" style="font-size: 1.15rem; padding: 0.75rem;">{v_label.upper()}</div>', unsafe_allow_html=True)
                st.metric("Compound Score", f"{v_compound:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_p2:
                st.markdown('<div class="dark-card" style="text-align: center;">', unsafe_allow_html=True)
                st.markdown("##### TextBlob (Polarity-Based)")
                c_style = "result-positive" if tb_label == "positive" else ("result-negative" if tb_label == "negative" else "result-neutral")
                st.markdown(f'<div class="result-container {c_style}" style="font-size: 1.15rem; padding: 0.75rem;">{tb_label.upper()}</div>', unsafe_allow_html=True)
                st.metric("Polarity Score", f"{tb_polarity:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_p3:
                st.markdown('<div class="dark-card" style="text-align: center;">', unsafe_allow_html=True)
                st.markdown("##### Logistic Regression")
                if lr_model is not None:
                    c_style = "result-positive" if ml_label == "positive" else ("result-negative" if ml_label == "negative" else "result-neutral")
                    st.markdown(f'<div class="result-container {c_style}" style="font-size: 1.15rem; padding: 0.75rem;">{ml_label.upper()}</div>', unsafe_allow_html=True)
                    st.metric("LR Confidence", f"{ml_conf * 100:.2f}%")
                else:
                    st.warning("Train model first.")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_p4:
                st.markdown('<div class="dark-card" style="text-align: center;">', unsafe_allow_html=True)
                st.markdown("##### Naive Bayes")
                if nb_model is not None:
                    c_style = "result-positive" if nb_label == "positive" else ("result-negative" if nb_label == "negative" else "result-neutral")
                    st.markdown(f'<div class="result-container {c_style}" style="font-size: 1.15rem; padding: 0.75rem;">{nb_label.upper()}</div>', unsafe_allow_html=True)
                    st.metric("NB Confidence", f"{nb_conf * 100:.2f}%")
                else:
                    st.warning("Train model first.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Word-level Attribution
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🕵️ Interactive Word-Level Sentiment Attribution")
            st.write("Hover over words to see their attribution direction. Green indicates positive terms, while Red indicates negative terms.")
            
            attribution_source = st.selectbox(
                "Attribution Weights Source",
                ["VADER Lexicon Model", "TextBlob Lexicon Model", "Custom Logistic Regression ML Model", "Custom Naive Bayes ML Model"]
            )
            
            # Tokenize original words preserving spacing
            words_and_spaces = re.split(r'(\s+|[^\w\s\-\'])', custom_tweet)
            html_parts = []
            word_weights = []
            
            for part in words_and_spaces:
                if re.match(r'^[\w\-\']+$', part):
                    part_clean = clean_tweet_text(part)
                    
                    if attribution_source == "VADER Lexicon Model":
                        weight = vader_analyzer.lexicon.get(part.lower(), 0.0) / 4.0 # Normalize VADER [-4, 4] to [-1, 1] range roughly
                    elif attribution_source == "TextBlob Lexicon Model":
                        weight = TextBlob(part).sentiment.polarity
                    elif attribution_source == "Custom Logistic Regression ML Model":
                        if lr_model is not None and vectorizer is not None and part_clean in vectorizer.vocabulary_:
                            vocab_idx = vectorizer.vocabulary_[part_clean]
                            # Let's subtract negative coefficient from positive coefficient to get direction
                            # Classes: ['negative', 'neutral', 'positive']
                            # coef_[0] corresponds to negative class, coef_[2] to positive class
                            weight = lr_model.coef_[2][vocab_idx] - lr_model.coef_[0][vocab_idx]
                        else:
                            weight = 0.0
                    elif attribution_source == "Custom Naive Bayes ML Model":
                        if nb_model is not None and vectorizer is not None and part_clean in vectorizer.vocabulary_:
                            vocab_idx = vectorizer.vocabulary_[part_clean]
                            # Let's subtract negative coefficient from positive coefficient to get direction
                            # Classes: ['negative', 'neutral', 'positive']
                            # feature_log_prob_[0] is negative, feature_log_prob_[2] is positive
                            weight = nb_model.feature_log_prob_[2][vocab_idx] - nb_model.feature_log_prob_[0][vocab_idx]
                        else:
                            weight = 0.0
                    else:
                        weight = 0.0
                            
                    if weight > 0.02:
                        opacity = min(0.9, 0.15 + abs(weight) * 0.8)
                        html_parts.append(
                            f'<span class="word-highlight word-positive" style="background-color: rgba(16, 185, 129, {opacity:.2f});" title="Positive impact: {weight:.4f}">{part}</span>'
                        )
                        word_weights.append((part, weight))
                    elif weight < -0.02:
                        opacity = min(0.9, 0.15 + abs(weight) * 0.8)
                        html_parts.append(
                            f'<span class="word-highlight word-negative" style="background-color: rgba(239, 68, 68, {opacity:.2f});" title="Negative impact: {weight:.4f}">{part}</span>'
                        )
                        word_weights.append((part, weight))
                    else:
                        html_parts.append(part)
                else:
                    html_parts.append(part)
                    
            highlighted_html = "".join(html_parts)
            
            st.markdown(
                f'<div style="background-color:#F9FAFB; border-radius:10px; padding:1.25rem; border:1px solid #E5E7EB; line-height:1.75; font-size:1.15rem; margin-bottom: 1.5rem;">{highlighted_html}</div>', 
                unsafe_allow_html=True
            )
            
            # Plot attribution contributors
            if word_weights:
                word_weights = sorted(word_weights, key=lambda x: abs(x[1]), reverse=True)[:10]
                words, weights = zip(*word_weights)
                
                fig, ax = plt.subplots(figsize=(8, 3))
                colors = ['#10B981' if w > 0 else '#EF4444' for w in weights]
                sns.barplot(x=list(weights), y=list(words), palette=colors, ax=ax)
                ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
                ax.set_title("Attribution Weights per Vocabulary Term", fontsize=10, fontweight='bold')
                ax.set_xlabel("Impact Score", fontsize=8)
                ax.tick_params(labelsize=8)
                plt.tight_layout()
                
                col_plot, col_legend = st.columns([3, 1])
                with col_plot:
                    st.pyplot(fig)
                with col_legend:
                    st.markdown("##### Chart Legend")
                    st.markdown(
                        """
                        - **Green (Positive)**: Terms driving prediction towards **Positive**.
                        - **Red (Negative)**: Terms driving prediction towards **Negative**.
                        - Values indicate the exact feature weights of the active model.
                        """
                    )
            else:
                st.info("No strong vocabulary keywords were found to attribute sentiment weights.")
            st.markdown('</div>', unsafe_allow_html=True)


# --- TAB 3: Preprocessing Pipeline ---
with tab_nlp:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Step-by-Step Natural Language Processing (NLP) Pipeline")
    st.write("Understand the transformations applied to tweets before they are evaluated by the machine learning model:")
    
    test_pipeline_text = st.text_input("Test Preprocessing Pipeline on custom text:", "Wow!! Checkout this link: https://t.co/xyz123 @Tesla is the best product! 🚗🚀 #awesome")
    
    # 1. Input
    st.markdown("##### 1. Raw Tweet Text")
    st.info(test_pipeline_text)
    
    # 2. Lowercase
    step1 = test_pipeline_text.lower()
    st.markdown("##### 2. Lowercase Conversion")
    st.code(step1)
    
    # 3. URL removal
    step2 = re.sub(r"https?://\S+|www\.\S+", "", step1)
    st.markdown(r"##### 3. URL Removal (Regex filter: `https?://\S+|www\.\S+`)")
    st.code(step2)
    
    # 4. Handle removal
    step3 = re.sub(r"@[A-Za-z0-9_]+", "", step2)
    st.markdown("##### 4. Twitter User Handle Removal (Regex filter: `@[A-Za-z0-9_]+`)")
    st.code(step3)
    
    # 5. Punctuation removal
    step4 = re.sub(f"[{string.punctuation}]", " ", step3)
    step4 = re.sub(r"\d+", " ", step4)
    st.markdown("##### 5. Punctuation & Digits Removal")
    st.code(step4)
    
    # 6. Stopwords removal
    tokens = step4.split()
    tokens_stopped = [w for w in tokens if w not in STOPWORDS]
    st.markdown("##### 6. Tokenization & English Stopwords Filtering")
    st.write(f"*Filtered vocabulary size: {len(STOPWORDS)} stopwords*")
    st.code(tokens_stopped)
    
    # 7. Lemmatization
    if USE_NLTK:
        tokens_lemmed = [lemmatizer.lemmatize(w) for w in tokens_stopped]
        desc = "Lemmatization (WordNet Lemmatizer)"
    else:
        tokens_lemmed = tokens_stopped
        desc = "Stopword Filtering (Lemmatizer unavailable)"
        
    st.markdown(f"##### 7. Word Lemmatization (Root Reduction)")
    st.code(tokens_lemmed)
    
    # 8. Vectorization
    st.markdown("##### 8. TF-IDF Fit check")
    if tokens_lemmed and vectorizer is not None:
        matched = []
        for t in tokens_lemmed:
            if t in vectorizer.vocabulary_:
                matched.append(f"'{t}' (Index: {vectorizer.vocabulary_[t]})")
        if matched:
            st.success(f"Matched Vocabulary Features in Classifier space: {', '.join(matched)}")
        else:
            st.warning("No tokens matched the 2500 training vocabulary features.")
    else:
        st.warning("Vectorizer models not loaded or empty final tokens.")
        
    st.markdown('</div>', unsafe_allow_html=True)


# --- TAB 4: ML Model Training ---
with tab_ml:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Machine Learning Model Training Center")
    st.write("Train, evaluate, and inspect Scikit-Learn classifiers on the generated Twitter sentiment corpus.")
    
    if ml_metrics is not None:
        st.success("✅ Models trained and ready on disk!")
        
        # Display Metrics side by side
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown('<div class="dark-card">', unsafe_allow_html=True)
            st.markdown("##### 📈 Logistic Regression performance")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{ml_metrics['lr_acc'] * 100:.2f}%")
            c2.metric("Precision", f"{ml_metrics['lr_prec'] * 100:.2f}%")
            c3.metric("Recall", f"{ml_metrics['lr_rec'] * 100:.2f}%")
            c4.metric("F1-Score", f"{ml_metrics['lr_f1'] * 100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_m2:
            st.markdown('<div class="dark-card">', unsafe_allow_html=True)
            st.markdown("##### 📐 Multinomial Naive Bayes performance")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{ml_metrics['nb_acc'] * 100:.2f}%")
            c2.metric("Precision", f"{ml_metrics['nb_prec'] * 100:.2f}%")
            c3.metric("Recall", f"{ml_metrics['nb_rec'] * 100:.2f}%")
            c4.metric("F1-Score", f"{ml_metrics['nb_f1'] * 100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown(f"**Dataset Training stats**: Training split: `{ml_metrics['train_size']}` tweets | Evaluation split: `{ml_metrics['test_size']}` tweets")
        
        # Display coefficients table
        if lr_model is not None and vectorizer is not None:
            st.markdown("#### 📊 Logistic Regression Vocabulary Coefficients")
            st.write("These coefficients show which words have the strongest impact on positive and negative predictions in the trained Logistic Regression model:")
            
            # Map coefficients to words
            feature_names = vectorizer.get_feature_names_out()
            # Positive Coefficients: Class 2
            pos_coefs = lr_model.coef_[2]
            # Negative Coefficients: Class 0
            neg_coefs = lr_model.coef_[0]
            
            coef_df = pd.DataFrame({
                "Word": feature_names,
                "Positive Coefficient": pos_coefs,
                "Negative Coefficient": neg_coefs
            })
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("##### 🟢 Top Positive Driver Words")
                st.dataframe(
                    coef_df.sort_values(by="Positive Coefficient", ascending=False).head(10)[["Word", "Positive Coefficient"]],
                    use_container_width=True
                )
            with col_t2:
                st.markdown("##### 🔴 Top Negative Driver Words")
                st.dataframe(
                    coef_df.sort_values(by="Negative Coefficient", ascending=False).head(10)[["Word", "Negative Coefficient"]],
                    use_container_width=True
                )
    else:
        st.warning("⚠️ No model files found on disk. Please trigger the training command.")
        
    st.markdown("---")
    
    if st.button("🔄 Retrain Models Now", type="primary"):
        with st.spinner("Executing pipeline (generate_dataset.py ➔ train_models.py)..."):
            try:
                # Run generator
                subprocess.run([sys.executable, "generate_dataset.py"], check=True)
                # Run training
                subprocess.run([sys.executable, "train_models.py"], check=True)
                st.success("Retrained successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error during retraining: {str(e)}")
            
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 0.85rem;'>Twitter Sentiment Analysis | OutriX AI Internship Portfolio Project</p>", unsafe_allow_html=True)
