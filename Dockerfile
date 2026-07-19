# Multi-stage Dockerfile for CTC-Detect
# Stage 1: Builder - install dependencies and compile any native extensions
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime AS builder

WORKDIR /workspace

# System dependencies for building
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers==4.41.0 \
    peft==0.11.0 \
    accelerate==0.30.0 \
    datasets==2.19.0 \
    scanpy==1.10.1 \
    anndata==0.10.7 \
    scikit-learn==1.4.2 \
    umap-learn==0.5.6 \
    matplotlib==3.8.4 \
    seaborn==0.13.2 \
    pyensembl==2.3.12 \
    loompy==3.0.7 \
    tdigest==0.5.2.2 \
    mygene==3.2.2 \
    jupyter==1.0.0 \
    ipykernel==6.29.4 \
    typer==0.12.0 \
    rich==13.7.0 \
    pydantic==2.7.0 \
    pydantic-settings==2.3.0 \
    huggingface_hub==0.36.0

# Install pyensembl reference genome
RUN pyensembl install --release 109 --species homo_sapiens

# Clone Geneformer
RUN git lfs install && \
    git clone https://huggingface.co/ctheodoris/Geneformer /workspace/Geneformer && \
    pip install --no-deps /workspace/Geneformer


# Stage 2: Runtime - minimal image with only runtime dependencies
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime AS runtime

WORKDIR /workspace

# Install only runtime system dependencies (no build tools)
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /opt/conda /opt/conda

# Copy Geneformer from builder
COPY --from=builder /workspace/Geneformer /workspace/Geneformer

# Copy application code
COPY . /workspace/ctc-detect
WORKDIR /workspace/ctc-detect

# Install package in development mode
RUN pip install -e . --no-deps

# Verify all imports work
RUN python -c "
import transformers, peft, accelerate, datasets, scanpy, torch, sklearn;
from peft import LoraConfig, get_peft_model, TaskType;
from transformers import AutoModelForSequenceClassification;
import ctcdetect;
print('ALL IMPORTS OK');
print('CTC-Detect version:', ctcdetect.__version__);
print('CUDA:', torch.cuda.is_available());
print('Torch:', torch.__version__);
print('Transformers:', transformers.__version__);
print('PEFT:', peft.__version__)
"

# Default command: show help
ENTRYPOINT ["ctc-detect"]
CMD ["--help"]