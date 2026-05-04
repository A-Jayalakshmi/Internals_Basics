import pandas as pd
import json

# Load training data
train_df = pd.read_csv("data/training_data.csv")

# Compute training means
train_mean_cl = train_df["content_length_min"].mean()
train_mean_ut = train_df["user_tenure_months"].mean()

# Load prediction logs
logs = []
with open("logs/predictions.jsonl", "r") as f:
    for line in f:
        logs.append(json.loads(line))

# Convert logs to DataFrame
live_df = pd.DataFrame([entry["input"] for entry in logs])

# Compute live means
live_mean_cl = live_df["content_length_min"].mean()
live_mean_ut = live_df["user_tenure_months"].mean()

# Thresholds (given in question)
threshold_cl = 29.51
threshold_ut = 15.12

# Calculate shifts
shift_cl = abs(live_mean_cl - train_mean_cl)
shift_ut = abs(live_mean_ut - train_mean_ut)

alerts = []

# Check drift for content_length_min
if shift_cl > threshold_cl:
    alerts.append({
        "feature": "content_length_min",
        "train_mean": round(train_mean_cl, 2),
        "live_mean": round(live_mean_cl, 2),
        "shift": round(shift_cl, 2),
        "threshold": threshold_cl,
        "status": "ALERT"
    })

# Check drift for user_tenure_months
if shift_ut > threshold_ut:
    alerts.append({
        "feature": "user_tenure_months",
        "train_mean": round(train_mean_ut, 2),
        "live_mean": round(live_mean_ut, 2),
        "shift": round(shift_ut, 2),
        "threshold": threshold_ut,
        "status": "ALERT"
    })

# Final result
result = {
    "total_predictions": len(logs),
    "mean_prediction": round(
        sum(entry["prediction"] for entry in logs) / len(logs), 4
    ),
    "drift_detected": len(alerts) > 0,
    "alerts": alerts
}

# Save output
with open("results/step4_s5.json", "w") as f:
    json.dump(result, f, indent=4)

print(" Monitoring completed")
print(json.dumps(result, indent=4))