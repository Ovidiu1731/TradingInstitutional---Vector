# Use Python 3.11 for better performance
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools (needed for some Python packages)
    build-essential \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose the port
EXPOSE 8080

# Run the application
# Note: Using app:app because your FastAPI instance is in app.py
# This is now a text-based trading education assistant
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]