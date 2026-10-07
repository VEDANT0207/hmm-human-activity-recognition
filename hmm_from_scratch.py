import numpy as np

class GaussianHMMFromScratch:
    """
    Gaussian Hidden Markov Model implemented completely from scratch.
    - Training: Supervised Maximum Likelihood Estimation (MLE) with Laplace smoothing
    - Inference: Viterbi Algorithm implemented strictly in Log-Space (prevents underflow)
    """
    def __init__(self, n_states=3, smoothing=1e-6):
        self.n_states = n_states
        self.smoothing = smoothing
        
        # HMM Parameters: lambda = (pi, A, mu, sigma)
        self.pi = np.zeros(n_states)
        self.A = np.zeros((n_states, n_states))
        self.means = np.zeros(n_states)
        self.variances = np.zeros(n_states)

    def fit(self, sequences_obs, sequences_states):
        """
        Supervised parameter estimation using counts and Gaussian MLE.
        """
        n_seqs = len(sequences_states)
        
        # 1. Initial State Distribution (pi)
        start_counts = np.zeros(self.n_states)
        for s_seq in sequences_states:
            start_counts[s_seq[0]] += 1
        self.pi = (start_counts + self.smoothing) / np.sum(start_counts + self.smoothing)

        # 2. State Transition Matrix (A)
        trans_counts = np.zeros((self.n_states, self.n_states))
        for s_seq in sequences_states:
            for t in range(len(s_seq) - 1):
                from_s = s_seq[t]
                to_s = s_seq[t + 1]
                trans_counts[from_s, to_s] += 1
                
        # Row-normalize with Laplace smoothing
        self.A = (trans_counts + self.smoothing) / (
            np.sum(trans_counts + self.smoothing, axis=1, keepdims=True)
        )

        # 3. Gaussian Emission Parameters (means, variances)
        all_obs_flat = np.concatenate(sequences_obs)
        all_states_flat = np.concatenate(sequences_states)

        for s in range(self.n_states):
            state_obs = all_obs_flat[all_states_flat == s]
            if len(state_obs) > 0:
                self.means[s] = np.mean(state_obs)
                self.variances[s] = np.var(state_obs) + 1e-4  # ensure strictly positive variance
            else:
                self.means[s] = 0.0
                self.variances[s] = 1.0

        print("--- HMM Model Parameters Learned (Supervised MLE) ---")
        print(f"Initial State Probabilities (pi):\n{self.pi}")
        print(f"\nTransition Matrix (A):\n{np.round(self.A, 4)}")
        print(f"\nGaussian Means (mu):\n{np.round(self.means, 4)}")
        print(f"Gaussian Variances (sigma^2):\n{np.round(self.variances, 4)}")

    def _log_gaussian_pdf(self, x, mean, var):
        """Computes log N(x; mean, var) safely without division-by-zero."""
        return -0.5 * np.log(2.0 * np.pi * var) - ((x - mean) ** 2) / (2.0 * var)

    def viterbi(self, obs_seq):
        """
        Log-Space Viterbi Algorithm.
        Decodes the globally optimal hidden state sequence for a given observation sequence.
        """
        T = len(obs_seq)
        N = self.n_states

        log_pi = np.log(self.pi)
        log_A = np.log(self.A)

        # V[t, j] stores the max log-probability ending in state j at time t
        V = np.zeros((T, N))
        # Backpointer matrix to track best prior state
        B = np.zeros((T, N), dtype=int)

        # Step 1: Initialization at t = 0
        for j in range(N):
            log_emission = self._log_gaussian_pdf(obs_seq[0], self.means[j], self.variances[j])
            V[0, j] = log_pi[j] + log_emission

        # Step 2: Recursion for t = 1 ... T-1
        for t in range(1, T):
            for j in range(N):
                log_emission = self._log_gaussian_pdf(obs_seq[t], self.means[j], self.variances[j])
                
                # Compare all possible previous states i
                transition_scores = V[t - 1, :] + log_A[:, j]
                best_prev_state = np.argmax(transition_scores)
                
                V[t, j] = transition_scores[best_prev_state] + log_emission
                B[t, j] = best_prev_state

        # Step 3: Termination at t = T - 1
        best_last_state = np.argmax(V[T - 1, :])

        # Step 4: Backtracking
        best_path = np.zeros(T, dtype=int)
        best_path[T - 1] = best_last_state
        for t in range(T - 2, -1, -1):
            best_path[t] = B[t + 1, best_path[t + 1]]

        return best_path

    def predict_sequences(self, sequences_obs):
        """Runs Viterbi on a batch of observation sequences."""
        return [self.viterbi(seq) for seq in sequences_obs]
