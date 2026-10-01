# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy entire project
COPY . .

# Expose port for Flask API
EXPOSE 5000

# Run the API when container starts
CMD ["python", "sentiment_api.py"]