FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pin all ML dependencies to known-compatible versions
# These match the training environment (transformers 4.41.0, peft 0.11.0, torch 2.3.x)
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
    ipykernel==6.29.4

# Install pyensembl reference genome
RUN pyensembl install --release 109 --species homo_sapiens

# Clone Geneformer
RUN git lfs install && \
    git clone https://huggingface.co/ctheodoris/Geneformer /workspace/Geneformer && \
    pip install --no-deps /workspace/Geneformer

# Verify all imports work
RUN python -c "\
import transformers, peft, accelerate, datasets, scanpy, torch, sklearn; \
from peft import LoraConfig, get_peft_model, TaskType; \
from transformers import AutoModelForSequenceClassification; \
print('ALL IMPORTS OK'); \
print('CUDA:', torch.cuda.is_available()); \
print('Torch:', torch.__version__); \
print('Transformers:', transformers.__version__); \
print('PEFT:', peft.__version__)"

COPY . /workspace/ctc-detect
WORKDIR /workspace/ctc-detect

CMD ["python", "src/model/finetune.py"]
