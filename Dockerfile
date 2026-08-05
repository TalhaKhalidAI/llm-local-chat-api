# Dockerfile
FROM ubuntu:22.04

# ============================================
# 1. Install System Dependencies + Vulkan
# ============================================
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    libsqlite3-dev \
    dnsutils \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# 2. Set Python 3.10 as default
# ============================================
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# ============================================
# 3. Install Python dependencies
# ============================================
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir python-dotenv uvicorn[standard]

# ============================================
# 4. Copy the entire application
# ============================================
COPY . .

# ============================================
# 5. Create directories
# ============================================
RUN mkdir -p /logs /models

# ============================================
# 6. Environment variables
# ============================================
ENV PYTHONUNBUFFERED=1
ENV LOG_FILEPATH=/logs
ENV LLM_MODEL_PATH=/models
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json

# ============================================
# 7. Expose ports
# ============================================
EXPOSE 8000

# ============================================
# 8. Health check
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ============================================
# 9. Start the application
# ============================================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]