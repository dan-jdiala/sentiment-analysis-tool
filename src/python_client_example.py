import requests
import json

BASE_URL = "http://localhost:5000/api/v1"

# Analyze a review
def analyze_review(text, username="user123"):
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={
            "text": text,
            "username": username,
            "save_to_db": True
        }
    )
    return response.json()

# Get statistics
def get_stats(days=30):
    response = requests.get(f"{BASE_URL}/statistics?days={days}")
    return response.json()

# Search reviews
def search(keyword):
    response = requests.get(f"{BASE_URL}/search?q={keyword}")
    return response.json()

# Usage
result = analyze_review("This product is amazing!")
print(f"Sentiment: {result['analysis']['sentiment']}")
print(f"Review ID: {result['review_id']}")