# Dockerfile
FROM python:3.10-slim

# ============================================
# 1. System Dependencies + Vulkan + SPIRV
# ============================================
RUN apt-get update && apt-get install -y --fix-missing \
    curl wget git \
    build-essential cmake ninja-build pkg-config \
    libvulkan1 libvulkan-dev vulkan-tools mesa-vulkan-drivers \
    spirv-headers spirv-tools \
    glslc \
    libsqlite3-dev dnsutils iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Verify glslc exists
RUN glslc --version

# ============================================
# 2. Install Python dependencies
# ============================================
WORKDIR /app
COPY requirements.txt ./
RUN grep -v "llama-cpp-python" requirements.txt > requirements.no-llama.txt && \
    pip install --no-cache-dir --timeout=100 -r requirements.no-llama.txt

# ============================================
# 3. Install llama-cpp-python with Vulkan support
# ============================================
RUN CMAKE_ARGS="-DGGML_VULKAN=ON" pip install --no-cache-dir llama-cpp-python

# ============================================
# 4. Copy the rest of the application
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
# 7. Expose and run
# ============================================
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]