FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables to ensure offline mode
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Run the main script
CMD ["python", "process.py", "--input_json", "challenge1b_input.json"]
