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
