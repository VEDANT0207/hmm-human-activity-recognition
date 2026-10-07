# Human Activity Recognition (HAR) using Hidden Markov Models (HMM)

This project implements a complete, end-to-end Hidden Markov Model pipeline from scratch in Python to classify human activities from raw smartphone accelerometer data.

---

## 1. Project Overview & Real-World Dataset
* **Dataset:** WISDM (Wireless Sensor Data Mining) Smartphone Activity Dataset (Fordham University).
* **Device:** Android smartphone carried in the user's front pants pocket.
* **Sampling Rate:** 20 Hz (20 readings/second = 50ms interval).
* **Total Samples:** 1,086,465 sensor measurements collected across 36 human subjects.
* **Subject-Independent Validation:**
  * **Training Set:** Subjects 1 through 28 (8,216 continuous sequences of length 100).
  * **Test Set:** Subjects 29 through 36 (2,630 continuous sequences of length 100).
  * *Evaluated strictly on unseen human subjects.*

---

## 2. Model Formulation: $\lambda = (\pi, A, B)$

### Hidden States ($S$)
1. **Stationary ($s_0$):** Sitting and Standing.
2. **Walking ($s_1$):** Walking, Upstairs, Downstairs.
3. **Jogging ($s_2$):** Jogging.

### Continuous Observations ($O$)
Invariant 3D acceleration magnitude (Euclidean norm):
$$r_t = \sqrt{x_t^2 + y_t^2 + z_t^2}$$

### Learned Parameters (Supervised MLE)
* **Initial Probabilities ($\pi$):**
  $$\pi = [0.0898, \ 0.5971, \ 0.3130]$$
* **Transition Matrix ($A$):**
  $$A \approx \begin{bmatrix} 0.9999 & 0.0001 & 0.0000 \\ 0.0000 & 0.9999 & 0.0001 \\ 0.0000 & 0.0001 & 0.9999 \end{bmatrix}$$
  *(Strong diagonal dominance reflects physical state persistence in real human activity).*
* **Gaussian Emission Parameters:**
  * **Stationary:** $\mu_0 = 9.84 \text{ m/s}^2$ (Earth's gravity), $\sigma_0^2 = 0.14$ (near-zero variance).
  * **Walking:** $\mu_1 = 11.18 \text{ m/s}^2$, $\sigma_1^2 = 21.34$ (moderate cyclic variance).
  * **Jogging:** $\mu_2 = 13.62 \text{ m/s}^2$, $\sigma_2^2 = 51.87$ (high-impact variance).

---

## 3. Results & Benchmark Comparison

| Model | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| **From-Scratch Log-Viterbi HMM** | **90.50%** | **0.91** | **0.93** | **0.92** |
| **Industry Benchmark (`hmmlearn`)** | **90.50%** | **0.91** | **0.93** | **0.92** |

* **Algorithm Agreement:** **100.00% exact match** between our custom Log-Viterbi implementation and `hmmlearn.hmm.GaussianHMM`.
* **Numerical Stability:** Implemented in log-space to guarantee zero floating-point underflow on long time-series.

---

## 4. Project Files in `R:\HMM_model_HAR`
* [`data_loader.py`](data_loader.py): Parses raw text, cleans semicolons, computes magnitude, maps states, and creates subject-split sequences.
* [`hmm_from_scratch.py`](hmm_from_scratch.py): Pure Python/NumPy implementation of supervised MLE and Log-Viterbi decoding.
* [`main_pipeline.py`](main_pipeline.py): Orchestrates end-to-end execution, benchmark validation, and generates performance plots.
* [`confusion_matrix.png`](confusion_matrix.png): Multi-class confusion matrix on unseen test subjects.
* [`sample_sequence_decoding.png`](sample_sequence_decoding.png): Continuous time-series sensor plot with ground truth vs. Viterbi predictions.

---

## 5. How to Re-Run
```powershell
cd R:\HMM_model_HAR
python main_pipeline.py
```
