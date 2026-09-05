# NeoRed: A Knowledge-Logic-Alignment Multimodal Large Language Model for Neonatal Respiratory Disease Diagnosis

[![Paper](https://img.shields.io/badge/arXiv-2609.03527-b31b1b.svg)](https://arxiv.org/abs/2609.03527)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.8-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **NeoRed**, a multimodal large language model designed for neonatal respiratory disease diagnosis from chest X-rays and multidimensional clinical context.

> **Release status:** The research code is available. Model weights and the NeoCXR/NeoCXR-EV datasets are not included in this repository. The datasets will be available upon application, subject to the required clinical data-use and ethics approvals.

## Overview

Existing medical MLLMs are predominantly trained on adult data and do not fully exploit the clinical context needed for neonatal diagnosis. NeoRed jointly models chest X-rays with developmental factors, perinatal risks, and physiological status to generate structured diagnostic reports.

NeoRed introduces a **Knowledge-Logic-Alignment (KLA)** framework with three components:

- **Knowledge Prior Injection (KPI):** injects neonatologist-inspired diagnostic priors into multimodal representations.
- **Diagnostic Logic Constraint (DLC):** aligns generated reports with multimodal diagnostic logic.
- **Visual Semantic Alignment (VSA):** establishes semantic correspondence between visual features and imaging conclusions.

## Main Results

On the NeoCXR benchmark, NeoRed achieves:

| Metric | Score |
| --- | ---: |
| ROUGE-L | 53.29% |
| Clinical Efficacy F1 | 65.19% |

The model also maintains competitive report-generation performance on the adult MIMIC-CXR and IU-Xray benchmarks. See the [paper](https://arxiv.org/abs/2609.03527) for full experimental details.

## Repository Structure

```text
NeoRed/
├── llava/                  # Model, training, serving, and evaluation code
├── scripts/
│   ├── pretrain.sh         # Multimodal projector pretraining
│   ├── finetune_lora.sh    # NeoRed LoRA fine-tuning
│   └── eval.sh             # Report generation and evaluation
└── pyproject.toml          # Python package and dependencies
```

## Installation

We recommend a Linux environment with CUDA-capable GPUs.

```bash
git clone https://github.com/yogurt-lyn/NeoRed.git
cd NeoRed
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The reference environment uses Python 3.8+, PyTorch 2.0.1, Transformers 4.31.0, and DeepSpeed 0.9.5. Please select CUDA-compatible PyTorch packages for your system when necessary.

## Data Preparation

The training pipeline expects a conversation-style annotation file and a corresponding image directory. A sample record follows this structure:

```json
{
  "image": "example.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nGenerate a diagnostic report from the image and clinical context."},
    {"from": "gpt", "value": "Reference diagnostic report."}
  ]
}
```

Before running the scripts, configure the dataset, image, base-model, vision-encoder, and output paths in `scripts/pretrain.sh`, `scripts/finetune_lora.sh`, and `scripts/eval.sh`. The checked-in scripts contain example paths from the authors' research environment and must be adapted to your machine.

## Training

Pretrain the multimodal projector:

```bash
bash scripts/pretrain.sh ./checkpoints 1 32 2
```

Fine-tune NeoRed with LoRA after configuring the paths in the script:

```bash
bash scripts/finetune_lora.sh
```

The fine-tuning script is configured for distributed training with DeepSpeed. Adjust the batch size, gradient accumulation, and distributed settings according to your hardware.

## Evaluation

After setting the query file, image folder, base model, and trained checkpoint paths:

```bash
bash scripts/eval.sh \
  /path/to/base-model \
  /path/to/neored-checkpoint \
  results/neored \
  neored-eval
```

The evaluation pipeline generates reports and computes radiology report-generation metrics using the utilities under `llava/eval/rrg_eval/`.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{liu2026neored,
  title        = {NeoRed: A Knowledge-Logic-Alignment Multimodal Large Language Model for Neonatal Respiratory Disease Diagnosis},
  author       = {Liu, Yinan and Xia, Hongtai and Xu, Haoran and Hong, Jiankang and Song, Jingkuan and Luo, Ye},
  year         = {2026},
  eprint       = {2609.03527},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2609.03527}
}
```

## Acknowledgements

This codebase builds on [LLaVA-Rad](https://github.com/microsoft/LLaVA-Rad). We thank the authors and the open-source community for their contributions.
