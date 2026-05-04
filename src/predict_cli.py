import argparse
import joblib
import json

# Load trained model
model = joblib.load("models/best_model.pkl")

# Argument parser
parser = argparse.ArgumentParser()

parser.add_argument("--content_length_min", type=float, required=True)
parser.add_argument("--user_tenure_months", type=int, required=True)
parser.add_argument("--is_premium", type=int, required=True)
parser.add_argument("--genre_score", type=int, required=True)

args = parser.parse_args()

# Prepare input
features = [[
    args.content_length_min,
    args.user_tenure_months,
    args.is_premium,
    args.genre_score
]]

# Predict
prediction = model.predict(features)[0]

# Output JSON (required format)
output = {
    "image_name": "streamcast-predictor",
    "image_tag": "v1",
    "base_image": "python:3.12-slim",
    "test_input": {
        "content_length_min": args.content_length_min,
        "user_tenure_months": args.user_tenure_months,
        "is_premium": args.is_premium,
        "genre_score": args.genre_score
    },
    "prediction": round(float(prediction), 4)
}

# Save result
with open("results/step2_s3.json", "w") as f:
    json.dump(output, f, indent=4)

# Print result
print(json.dumps(output, indent=4))