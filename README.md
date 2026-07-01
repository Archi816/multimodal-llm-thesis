# multimodal-llm-thesis
BLIP-3 fine-tuning with LoRA on VQAv2 benchmark; 70.1% accuracy (+9.8pp); HPC PERUN; NVIDIA H200; Bachelor thesis TUKE FEI 2026

# Introduction to Multimodal Large Language Models

Bachelor's thesis — Technical University of Košice, FEI, 2026  
Supervisor: prof. Ing. Peter Sinčák, CSc.

## Overview

This thesis provides a structured introduction to multimodal large language models (MLLMs), tracing their architectural evolution from unimodal transformer systems to unified vision-language architectures.

The practical part implements **BLIP-3** on the **HPC PERUN supercomputing cluster** (NVIDIA H200 GPU) at TUKE, applying **LoRA fine-tuning** on the **VQAv2** visual question answering benchmark.

## Results

| Metric | Base Model | Fine-tuned |
|--------|-----------|------------|
| Overall accuracy | 60.3% | **70.1%** |
| Numerical questions | 2.4% | **50.4%** |
| Parameters updated | — | 0.082% (LoRA) |

Improvement of **+9.8 percentage points**, statistically confirmed by non-overlapping 95% Wilson confidence intervals.

## Key Topics

- Shared embedding spaces, cross-modal attention, vision-language pre-training
- Architecture deep-dive: Flamingo, LLaVA, BLIP-3
- AnyRes vision token sampling & Perceiver Resampler token compression
- LoRA parameter-efficient fine-tuning
- Attention map visualisation across all 32 transformer layers

## Tech Stack

- Python · PyTorch
- BLIP-3 (xGen-MM)
- LoRA / PEFT
- VQAv2 dataset
- HPC PERUN · NVIDIA H200 GPU

## Keywords

BLIP-3 · Multimodal LLM · Visual Question Answering · LoRA · Parameter-efficient fine-tuning · HPC · VQAv2 · Cross-modal attention · Vision Transformer