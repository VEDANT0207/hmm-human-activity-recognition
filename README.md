# Human Activity Recognition (HAR) using Hidden Markov Models (HMM)

An Advanced AI project implementing a Gaussian Hidden Markov Model (HMM) from scratch in Python to recognize human physical activities from smartphone accelerometer data.

> 📌 **Primary Artifact**: Please refer directly to the Jupyter Notebook [`HMM_Human_Activity_Recognition.ipynb`](HMM_Human_Activity_Recognition.ipynb) for the full step-by-step implementation, mathematical explanations, exploratory visualizations, and benchmark results.

---

## Quick Summary
- **Dataset**: WISDM Smartphone Activity Dataset (~1.08 million accelerometer readings across 36 subjects).
- **Activities**: Stationary (Sitting/Standing), Walking (Walking/Stairs), Jogging.
- **Implementation**: Pure NumPy Gaussian HMM built from scratch (Supervised MLE parameter estimation + Log-Space Viterbi algorithm).
- **Validation**: Subject-Independent split (Train: Users 1–28, Test: Unseen Users 29–36).
- **Results**: **90.50% Test Accuracy** with a **100.00% exact match** against Python's official `hmmlearn` library.

---

## Repository Files
- [`HMM_Human_Activity_Recognition.ipynb`](HMM_Human_Activity_Recognition.ipynb) - Primary project notebook containing all code, explanations, and outputs.
- [`data_loader.py`](data_loader.py) - Data cleaning, magnitude feature calculation, and sequence generation.
- [`hmm_from_scratch.py`](hmm_from_scratch.py) - From-scratch Gaussian HMM model class.
- [`main_pipeline.py`](main_pipeline.py) - Execution & benchmark evaluation script.
- [`requirements.txt`](requirements.txt) - Dependency list.
