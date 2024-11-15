# Use an official Python runtime as a base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
COPY nltk_data.py .
RUN python nltk_data.py

# Set NLTK data path
ENV NLTK_DATA=/usr/local/share/nltk_data

# Expose the Streamlit default port
EXPOSE 8501

# Copy all project files into the container
COPY . .

# Command to run your Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
