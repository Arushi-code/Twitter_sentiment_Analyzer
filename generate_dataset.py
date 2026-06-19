import pandas as pd
import random
import os

positive = [
    "I love this product",
    "Amazing experience",
    "Excellent service",
    "Fantastic work",
    "Very happy with the results",
    "Highly recommended",
    "Best update ever",
    "Outstanding quality",
    "Great customer support",
    "This is awesome"
]

negative = [
    "I hate this product",
    "Terrible experience",
    "Worst service ever",
    "Very disappointed",
    "Poor quality",
    "This is awful",
    "Not recommended",
    "Completely useless",
    "Bad customer support",
    "Waste of money"
]

neutral = [
    "The meeting is tomorrow",
    "I am reading a book",
    "The weather is normal",
    "This is a standard report",
    "Task completed successfully",
    "Project update received",
    "The package arrived today",
    "Reviewing the document",
    "Data processing started",
    "Information has been shared"
]

tweets = []

for _ in range(1000):
    tweets.append({
        "text": random.choice(positive),
        "sentiment": "positive"
    })

for _ in range(1000):
    tweets.append({
        "text": random.choice(negative),
        "sentiment": "negative"
    })

for _ in range(1000):
    tweets.append({
        "text": random.choice(neutral),
        "sentiment": "neutral"
    })

random.shuffle(tweets)

os.makedirs("data", exist_ok=True)

df = pd.DataFrame(tweets)
df.to_csv("data/tweets.csv", index=False)

print("Dataset generated successfully!")
print("Total samples:", len(df))