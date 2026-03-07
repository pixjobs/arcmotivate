# Use the official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables for Gradio stability on Cloud Run
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_ANALYTICS_ENABLED="False"

# Expose the standard Cloud Run port
EXPOSE 8080

# Command to run the application
# Use the dynamic PORT env var if available (handled in app.py)
CMD ["python", "app.py"]
