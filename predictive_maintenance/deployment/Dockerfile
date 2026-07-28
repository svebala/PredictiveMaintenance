# Use a lightweight Python base image
FROM python:3.11-slim

# Install Python dependencies
WORKDIR /tmp
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security
RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy application files
COPY --chown=user . $HOME/app

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
