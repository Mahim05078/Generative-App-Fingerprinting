# Generative-App-Fingerprinting

An end-to-end project for attributing AI-generated images to the application that produced them, not just the underlying model.

This repository combines notebook-driven experiments, forensic feature learning, and publication-style reporting to study how application pipelines leave behind detectable visual fingerprints. The core idea is simple: two apps can share a generator, but still produce images with different residual, spectral, and post-processing traces.

## Project Snapshot

The project evaluates four complementary views of the attribution problem:

1. **Experiment 1A - Non-Visual Baseline**
	A logistic regression model trained on image metadata and basic statistics.
2. **Experiment 1B - Semantic Baselines**
	ConvNeXt-Tiny, EfficientNet-B0, and ViT-B/16 fine-tuned on the application labels.
3. **Experiment 2A - Residual Fingerprinting**
	A RemNet-style residual model designed to expose content-independent artifacts.
4. **Experiment 2B - Multimodal Fusion**
	A dual-branch model that combines residual features with compact frequency descriptors.

## What Makes It Interesting

- The dataset contains **323 images across 8 applications** with noticeable class imbalance and varied resolutions.
- The best residual model outperforms the semantic baseline, suggesting that application identity is encoded in low-level artifacts.
- Frequency fusion is not automatically beneficial, which makes the project useful for studying negative transfer in multimodal forensic models.
