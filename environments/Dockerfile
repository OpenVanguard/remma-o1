# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    cmake \
    build-essential \
    libgl1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install Python dependencies first (layer caching optimization)
COPY environments/requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir jupyterlab

# Install project-specific tools
RUN pip install --no-cache-dir \
    huggingface_hub \
    wandb

# Copy project files
COPY . .

# Environment variables
ENV PYTHONPATH=/app/src
ENV HF_HOME=/app/cache/huggingface
ENV WANDB_DIR=/app/cache/wandb

# Expose Jupyter port
EXPOSE 8888

# Data volumes
VOLUME /app/data
VOLUME /app/models
VOLUME /app/cache

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Entrypoint
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Default command (override when running)
CMD ["--NotebookApp.token=''", "--NotebookApp.password=''"]