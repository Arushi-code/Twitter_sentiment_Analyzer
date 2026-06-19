import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Create data directory if not exists
os.makedirs("data", exist_ok=True)

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Templates for generating tweets
positive_templates = [
    "Absolutely love the new features on {product}! It's a total game changer. 🚀 #innovation",
    "Had an amazing experience with the customer service today. Super helpful and friendly! 🌟",
    "So excited for the upcoming launch. Can't wait to see what they have built! #excited",
    "This is hands down the best investment I've made this year. High quality and worth every penny.",
    "Congrats to the team on a successful release! Hard work really pays off. 🎉 #achievement",
    "Feeling extremely productive today. Got so much done using the new {product} app!",
    "The new updates are incredible. Performance is smoother and the UI looks stunning. 😍",
    "Just finished watching the new movie. It was absolutely brilliant! 10/10 recommend.",
    "Wow, I'm genuinely impressed by the speed of this service. Kudos! 🙌",
    "Beautiful weather today, perfect day for a walk in the park. Life is good! ☀️",
    "Really happy with how this project turned out. Big thanks to everyone involved!",
    "Highly recommend checking out this new tech. It simplifies everything. #tech #cool",
    "Such a wholesome post. Made me smile! 😊",
    "The lecture today on AI was fascinating. So much to learn and explore! #AI",
    "Super clean codebase, very easy to integrate and start building. Great developer experience!",
    "Had a great workout session. Health is wealth! 💪 #fitness",
    "This new restaurant is superb. The food was delicious and the staff was so polite.",
    "Incredible performance by the engineers. Resolved the issue within minutes!",
    "So grateful for all the support from my followers. You guys are the best! ❤️",
    "Loving this song, on repeat all day. What a vibe!"
]

negative_templates = [
    "Terrible experience with the new update. The app keeps crashing every 5 minutes! 😡 #fail",
    "Customer service was extremely rude and unhelpful. Disappointed. #worstservice",
    "This product is way overpriced for the quality. Save your money, don't buy it.",
    "Such a waste of time. The tutorial was confusing and didn't solve my problem.",
    "Really frustrated with the network downtime. I lost hours of work today. 😭 #frustrated",
    "The new UI design is awful. It's so hard to navigate now. Bring back the old version!",
    "Extremely disappointed with the shipping delay. Still haven't received my order.",
    "This movie was a major letdown. The plot was weak and acting was mediocre. 1/10",
    "I regret buying this. It stopped working after just two days of use. Avoid! #scam",
    "Worst support response ever. They ignored my query for three days and then closed the ticket.",
    "Traffic is absolutely horrible today. Stuck in the same spot for an hour. 🚗💨",
    "So tired of these endless bugs. Does anybody even test this software before release?",
    "It's raining cats and dogs, and I forgot my umbrella. Today is not my day. 🌧️",
    "The service was incredibly slow. Waited 45 minutes just to get cold food.",
    "I don't understand how this product got good reviews. It's cheap and fragile.",
    "My account got locked for no reason. This platform is so annoying. #fixthis",
    "Extremely laggy experience on {product}. It makes my device overheat.",
    "The presentation was so boring. Almost fell asleep. 😴",
    "Another security breach? This is unacceptable. Moving my data elsewhere immediately.",
    "Very bad quality. The stitches are coming off already. I want a refund."
]

neutral_templates = [
    "Just read an interesting article about {product}. Interesting perspective. #news",
    "Does anyone know if the office is open tomorrow? Thanks.",
    "Comparing the specifications of these two models. Both have pros and cons.",
    "Attending a webinar on data science this afternoon. #learning",
    "The meeting has been rescheduled to 3 PM. Please check your emails.",
    "Currently testing the new features of {product}. Will write a review soon.",
    "A regular day at the office. Coffee in hand, ready to work.",
    "Checking the weather report for the weekend. Looks like it might rain.",
    "Can someone recommend a good book on machine learning?",
    "Just updated my system. No noticeable differences so far.",
    "Lunch time. Having a sandwich today.",
    "The store is open from 9 AM to 9 PM daily. Good to know.",
    "Flights have been delayed due to low visibility. Standard protocol.",
    "Here is the link to the documentation for {product}. Let me know if it helps.",
    "Taking a short break from coding. Back in 10 minutes.",
    "Interesting to see how market trends are shifting this quarter.",
    "Does this library support Python 3.10? Checking the release notes.",
    "The package arrived today. Standard packaging, intact.",
    "Currently reading a research paper on natural language processing.",
    "Please send the report by the end of the day. Thanks."
]

products = ["Twitter", "X App", "Tesla Bot", "ChatGPT", "Apple Watch", "Bitcoin Wallet", "Netflix", "VS Code", "Windows 11", "GitHub CoPilot"]
usernames = ["tech_guru", "code_ninja", "cryptorider", "design_pro", "daily_scribe", "globetrotter", "ai_enthusiast", "gamer_girl", "coffee_lover", "fit_life"]

def generate_tweet_data(num_samples=1500):
    tweets = []
    
    # Generate balanced classes: 500 positive, 500 negative, 500 neutral
    samples_per_class = num_samples // 3
    
    for sentiment_label, templates in [("positive", positive_templates), 
                                       ("negative", negative_templates), 
                                       ("neutral", neutral_templates)]:
        for _ in range(samples_per_class):
            template = random.choice(templates)
            product = random.choice(products)
            username = random.choice(usernames)
            
            # Format text
            text = template.replace("{product}", product)
            
            # Add random handles/mentions and links to simulate real tweets
            if random.random() < 0.4:
                text = f"@{random.choice(usernames)} " + text
            if random.random() < 0.3:
                text = text + f" Check details: https://t.co/{random.randint(100000, 999999)}"
                
            # Random likes, retweets, and dates
            likes = int(np.random.exponential(scale=150))
            retweets = int(likes * np.random.uniform(0.1, 0.4))
            
            # Date in the last 30 days
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            tweet_date = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
            
            tweets.append({
                "tweet_id": str(random.randint(10**17, 10**18 - 1)),
                "username": username,
                "text": text,
                "sentiment": sentiment_label,
                "created_at": tweet_date.strftime("%Y-%m-%d %H:%M:%S"),
                "likes": likes,
                "retweets": retweets
            })
            
    # Shuffle dataset
    random.shuffle(tweets)
    return pd.DataFrame(tweets)

print("Generating 1500 realistic tweets...")
df_tweets = generate_tweet_data(1500)
df_tweets.to_csv("data/tweets.csv", index=False)
print("Dataset successfully saved to data/tweets.csv!")
print("Class Distribution:\n", df_tweets["sentiment"].value_counts())
