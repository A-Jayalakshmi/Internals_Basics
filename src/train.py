import pandas as pd
import mlflow
import mlflow.sklearn
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load dataset
df = pd.read_csv("data/training_data.csv")

# Split features and target
X = df.drop("watch_time_min", axis=1)
y = df["watch_time_min"]

# Train-test split (as per question)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set MLflow experiment
mlflow.set_experiment("streamcast-watch-time-min")

results = []
best_rmse = float("inf")
best_model = None
best_model_name = ""

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        # Log parameters
        if name == "Ridge":
            mlflow.log_param("alpha", 1.0)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # Tag
        mlflow.set_tag("experiment_type", "baseline_comparison")

        # Save model artifact
        mlflow.sklearn.log_model(model, name)

        # Store results
        results.append({
            "name": name,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4)
        })

        # Track best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name

# Save best model locally
joblib.dump(best_model, "models/best_model.pkl")

# Create JSON output
output = {
    "experiment_name": "streamcast-watch-time-min",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": round(best_rmse, 4)
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print(" Task 1 completed successfully!")