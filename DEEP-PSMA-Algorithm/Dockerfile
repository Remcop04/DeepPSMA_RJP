FROM --platform=linux/amd64 python:3.11-slim

# Prevents Python output buffering
ENV PYTHONUNBUFFERED=1

# Install system dependencies that might be needed
RUN apt-get update && apt-get install -y \
    git \
    wget \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and nnunetv2 system-wide first (before creating user)
# This ensures they're in the system PATH
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir nnunetv2

# Create user and group
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# Switch to user and set working directory
USER user
WORKDIR /opt/app

# Set nnU-Net environment variables (critical!)
ENV NNUNET_RESULTS_FOLDER="/opt/app/resources/nnUNet_data/results"
ENV NNUNET_RAW_DATA_BASE="/opt/app/resources/nnUNet_data/raw" 
ENV NNUNET_PREPROCESSED="/opt/app/resources/nnUNet_data/preprocessed"

# Ensure both system and user bin directories are in PATH
ENV PATH="/usr/local/bin:/home/user/.local/bin:$PATH"

# Copy and install requirements
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user --no-cache-dir --requirement /opt/app/requirements.txt

# Copy resources directory (this should contain your nnU-Net models)
COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user postprocessing.py /opt/app/
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user DEEP_PSMA_Infer.py /opt/app/
COPY --chown=user:user deep_psma_utils.py /opt/app/
COPY --chown=user:user nnunet_config_paths.py /opt/app/


# Verify critical components are available (optional but helpful for debugging)
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "PyTorch not found"
RUN which nnUNetv2_predict || echo "WARNING: nnUNetv2_predict not found in PATH"
RUN python -c "import nnunetv2; print('nnunetv2 imported successfully')" || echo "WARNING: nnunetv2 import failed"

# Check if model directories exist
RUN ls -la /opt/app/resources/ || echo "WARNING: resources directory structure:"

ENTRYPOINT ["python", "inference.py"]