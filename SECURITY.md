# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report security vulnerabilities by opening a **private security advisory** on GitHub:

1. Go to the [Security tab](https://github.com/gabufle/ctc-detect/security) of this repository
2. Click "Report a vulnerability"
3. Fill in the details and submit

We will acknowledge receipt within 72 hours and provide a timeline for a fix.

## Security Considerations for Users

### Model Downloads
- Models are downloaded from Hugging Face Hub using `huggingface_hub.snapshot_download`
- Only download models from trusted sources (the default registry uses `ctheodoris/Geneformer*`)
- If using a private model with `HF_TOKEN`, ensure the token has **read-only** scope
- Never use a write-token in CI/CD logs or shared environments

### Data Privacy
- Input scRNA-seq data is processed locally — **no data leaves your machine**
- Model inference runs entirely on your hardware (CPU or GPU)
- Output files (probabilities, UMAP, reports) are written to your specified output directory only

### Dependencies
- Core ML dependencies (`torch`, `transformers`, `peft`, `accelerate`) are pinned to specific versions in `pyproject.toml` and `Dockerfile` for reproducibility
- CI runs `pip-audit` on every PR to detect known vulnerabilities
- Dependabot creates weekly PRs for dependency updates

### Container Security
- The `Dockerfile` uses `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` as base
- Runs as non-root user (UID 1000) by default in CI
- No secrets baked into the image

## Disclosure Timeline

| Phase | Timeline |
|-------|----------|
| Acknowledgment | ≤ 72 hours |
| Initial assessment | ≤ 7 days |
| Fix development | ≤ 30 days (typical) |
| Public disclosure | After fix released + 14 days |

We credit reporters in release notes unless anonymity is requested.