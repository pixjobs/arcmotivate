# Use the official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variable to bind Gradio to 0.0.0.0
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Expose the Gradio port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
