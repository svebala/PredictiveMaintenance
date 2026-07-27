FROM python:3.9

# Create non-root user
RUN useradd -m -u 1000 user

# Working directory
WORKDIR /app

# Copy project
COPY . .

# Install dependencies as root
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER user

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
