import os
import numpy as np
import pandas as pd

def load_and_preprocess_wisdm(raw_file_path):
    """
    Loads raw WISDM dataset, parses accelerometer lines, computes magnitude,
    and maps activities to 3 well-separated states:
      0: Stationary (Sitting, Standing)
      1: Walking (Walking, Upstairs, Downstairs)
      2: Jogging (Jogging)
    """
    print(f"Loading raw data from: {raw_file_path}")
    data = []
    with open(raw_file_path, 'r') as f:
        for line in f:
            cleaned = line.strip().rstrip(';')
            parts = cleaned.split(',')
            if len(parts) == 6:
                data.append(parts)

    columns = ['user', 'activity', 'timestamp', 'x', 'y', 'z']
    df = pd.DataFrame(data, columns=columns)

    for col in ['x', 'y', 'z']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['user'] = pd.to_numeric(df['user'], errors='coerce')
    df = df.dropna()

    # Feature Engineering: Compute 3D Acceleration Magnitude (Euclidean Norm)
    df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

    # 3-State Mapping
    activity_map = {
        'Sitting': 'Stationary',
        'Standing': 'Stationary',
        'Walking': 'Walking',
        'Upstairs': 'Walking',
        'Downstairs': 'Walking',
        'Jogging': 'Jogging'
    }
    df['activity_mapped'] = df['activity'].map(activity_map)
    df = df.dropna(subset=['activity_mapped'])

    state_to_idx = {'Stationary': 0, 'Walking': 1, 'Jogging': 2}
    idx_to_state = {0: 'Stationary', 1: 'Walking', 2: 'Jogging'}
    df['state'] = df['activity_mapped'].map(state_to_idx)

    print(f"Total valid samples loaded: {len(df):,}")
    print("Class distribution:")
    for state_name, count in df['activity_mapped'].value_counts().items():
        print(f"  - {state_name}: {count:,} samples ({count/len(df)*100:.1f}%)")

    return df, state_to_idx, idx_to_state

def create_sequences(df, seq_len=100, train_users_cutoff=28):
    """
    Segments data into sequences of length `seq_len` per user.
    Uses Subject-Independent split:
      - Train users: user_id <= train_users_cutoff
      - Test users:  user_id > train_users_cutoff
    """
    train_obs, train_states = [], []
    test_obs, test_states = [], []

    for user_id, user_group in df.groupby('user'):
        obs_vals = user_group['magnitude'].values
        state_vals = user_group['state'].values.astype(int)

        n_samples = len(obs_vals)
        n_seqs = n_samples // seq_len

        for i in range(n_seqs):
            start = i * seq_len
            end = start + seq_len
            seq_o = obs_vals[start:end]
            seq_s = state_vals[start:end]

            if user_id <= train_users_cutoff:
                train_obs.append(seq_o)
                train_states.append(seq_s)
            else:
                test_obs.append(seq_o)
                test_states.append(seq_s)

    train_obs = np.array(train_obs)
    train_states = np.array(train_states)
    test_obs = np.array(test_obs)
    test_states = np.array(test_states)

    print(f"\nSequence generation (Length = {seq_len}):")
    print(f"Train sequences: {len(train_obs)} (Users <= {train_users_cutoff})")
    print(f"Test sequences:  {len(test_obs)} (Users > {train_users_cutoff})")

    return train_obs, train_states, test_obs, test_states
