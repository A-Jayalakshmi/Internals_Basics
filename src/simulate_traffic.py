import requests
import random

URL = "http://localhost:9000/score"

# 35 normal requests
for _ in range(35):
    data = {
        "content_length_min": random.uniform(10, 180),
        "user_tenure_months": random.randint(1, 60),
        "is_premium": random.randint(0, 1),
        "genre_score": random.randint(1, 10)
    }
    requests.post(URL, json=data)

# 15 drifted requests (higher values)
for _ in range(15):
    data = {
        "content_length_min": random.uniform(200, 350),
        "user_tenure_months": random.randint(70, 120),
        "is_premium": random.randint(0, 1),
        "genre_score": random.randint(1, 10)
    }
    requests.post(URL, json=data)

print("✅ Traffic simulation completed")