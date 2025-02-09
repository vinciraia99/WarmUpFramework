FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application script and models
COPY Models ./Models

# Expose port
EXPOSE 5000
