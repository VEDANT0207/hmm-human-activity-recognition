import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from hmmlearn import hmm

from data_loader import load_and_preprocess_wisdm, create_sequences
from hmm_from_scratch import GaussianHMMFromScratch

def run_pipeline():
    raw_file = r'R:\HMM_model_HAR\WISDM_ar_v1.1\WISDM_ar_v1.1_raw.txt'
    
    # 1. Load & Preprocess
    print("\n" + "="*50)
    print("STEP 1: LOADING & PREPROCESSING DATASET")
    print("="*50)
    df, state_to_idx, idx_to_state = load_and_preprocess_wisdm(raw_file)
    target_names = [idx_to_state[i] for i in range(len(idx_to_state))]

    # 2. Sequence Construction (Subject-Independent Split)
    print("\n" + "="*50)
    print("STEP 2: CONSTRUCTING SEQUENCES (Subject Split)")
    print("="*50)
    train_obs, train_states, test_obs, test_states = create_sequences(
        df, seq_len=100, train_users_cutoff=28
    )

    # 3. Train From-Scratch HMM (Supervised MLE)
    print("\n" + "="*50)
    print("STEP 3: TRAINING HMM FROM SCRATCH (Supervised MLE)")
    print("="*50)
    scratch_hmm = GaussianHMMFromScratch(n_states=3)
    scratch_hmm.fit(train_obs, train_states)

    # 4. Predict using Custom Log-Viterbi
    print("\n" + "="*50)
    print("STEP 4: INFERENCE USING LOG-VITERBI (FROM SCRATCH)")
    print("="*50)
    print("Decoding test sequences with Viterbi algorithm...")
    y_pred_scratch = scratch_hmm.predict_sequences(test_obs)

    y_test_flat = np.concatenate(test_states)
    y_pred_scratch_flat = np.concatenate(y_pred_scratch)

    acc_scratch = accuracy_score(y_test_flat, y_pred_scratch_flat)
    print(f"\n>>> From-Scratch Log-Viterbi Accuracy: {acc_scratch * 100:.2f}% <<<")
    print("\nClassification Report (From Scratch):")
    print(classification_report(y_test_flat, y_pred_scratch_flat, target_names=target_names))

    # 5. Benchmark Comparison: hmmlearn Library
    print("\n" + "="*50)
    print("STEP 5: BENCHMARK WITH STANDARD HMMLearn")
    print("="*50)
    # Configure hmmlearn GaussianHMM with the supervised parameters for 1-to-1 comparison
    lib_hmm = hmm.GaussianHMM(n_components=3, covariance_type="diag", init_params="")
    lib_hmm.startprob_ = scratch_hmm.pi
    lib_hmm.transmat_ = scratch_hmm.A
    lib_hmm.means_ = scratch_hmm.means.reshape(-1, 1)
    lib_hmm.covars_ = scratch_hmm.variances.reshape(-1, 1)

    y_pred_lib = []
    for seq in test_obs:
        _, state_seq = lib_hmm.decode(seq.reshape(-1, 1), algorithm="viterbi")
        y_pred_lib.append(state_seq)

    y_pred_lib_flat = np.concatenate(y_pred_lib)
    acc_lib = accuracy_score(y_test_flat, y_pred_lib_flat)
    print(f"\n>>> hmmlearn Library Viterbi Accuracy: {acc_lib * 100:.2f}% <<<")

    agreement = np.mean(y_pred_scratch_flat == y_pred_lib_flat) * 100
    print(f">>> Agreement between Scratch Viterbi and Library Viterbi: {agreement:.2f}% <<<")

    # 6. Save Confusion Matrix Plot
    print("\n" + "="*50)
    print("STEP 6: GENERATING EVALUATION VISUALS")
    print("="*50)
    cm = confusion_matrix(y_test_flat, y_pred_scratch_flat)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("HMM Viterbi HAR - Confusion Matrix (Test Subjects)")
    plt.xlabel("Predicted Activity")
    plt.ylabel("True Activity")
    plt.tight_layout()
    cm_path = r'R:\HMM_model_HAR\confusion_matrix.png'
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # Plot sample sequence prediction vs true labels
    sample_idx = 0
    sample_obs = test_obs[sample_idx]
    sample_true = test_states[sample_idx]
    sample_pred = y_pred_scratch[sample_idx]

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(sample_obs, color='teal', label='Accel Magnitude (m/s²)')
    plt.title(f"Test Sequence #{sample_idx}: Sensor Observation vs Predicted Activity")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(sample_true, label='True Activity', color='green', linewidth=2)
    plt.plot(sample_pred, label='Viterbi Prediction', color='crimson', linestyle='--', linewidth=2)
    plt.yticks([0, 1, 2], target_names)
    plt.xlabel("Time Step (50ms increments)")
    plt.ylabel("Activity State")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    sample_plot_path = r'R:\HMM_model_HAR\sample_sequence_decoding.png'
    plt.savefig(sample_plot_path, dpi=300)
    plt.close()
    print(f"Saved sample sequence prediction plot to: {sample_plot_path}")
    print("\nPIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    run_pipeline()
