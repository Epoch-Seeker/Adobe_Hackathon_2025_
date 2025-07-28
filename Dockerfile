FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 poppler-utils && \
    pip install --no-cache-dir torch PyMuPDF transformers sentence-transformers numpy

# Copy project files
COPY process.py .
COPY input ./input

# Entry point
CMD ["python", "process.py", "--input_json", "./input/input.json"]
