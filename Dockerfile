FROM python:3.12-slim

WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
ENTRYPOINT ["python", "src/predict_cli.py"]