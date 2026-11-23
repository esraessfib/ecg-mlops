"""
ECG feature extraction from raw 12-lead signals.

For each recording, loads the waveform using wfdb, then extracts
statistical and heart-rate features per lead. The result is a flat
feature vector suitable for classical ML models.
"""

import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

DATA_DIR = Path("data/ptb-xl-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
SAMPLING_RATE = 100  # Hz — we use the 100 Hz recordings

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def load_signal(filename_lr: str) -> np.ndarray:
    """
    Load a raw ECG signal from a WFDB record file.

    Args:
        filename_lr: relative path to the record (without extension),
                     as stored in the 'filename_lr' column of the metadata.

    Returns:
        NumPy array of shape (n_samples, 12) — one column per lead.
    """
    record_path = DATA_DIR / filename_lr
    record = wfdb.rdrecord(str(record_path))
    return record.p_signal  # shape: (1000, 12) for 100 Hz 10-second recordings


def extract_statistical_features(signal: np.ndarray) -> dict:
    """
    Compute basic statistics for each of the 12 leads.

    Features per lead: mean, standard deviation, min, max, skewness, kurtosis.
    Total: 12 leads x 6 features = 72 features.
    """
    features = {}
    for i, lead in enumerate(LEADS):
        lead_signal = signal[:, i]
        features[f"{lead}_mean"]     = np.mean(lead_signal)
        features[f"{lead}_std"]      = np.std(lead_signal)
        features[f"{lead}_min"]      = np.min(lead_signal)
        features[f"{lead}_max"]      = np.max(lead_signal)
        features[f"{lead}_skew"]     = skew(lead_signal)
        features[f"{lead}_kurtosis"] = kurtosis(lead_signal)
    return features


def extract_hrv_features(signal: np.ndarray) -> dict:
    """
    Estimate heart rate variability (HRV) features from lead II R-peaks.

    Uses lead II (index 1) which has the clearest R-wave in most recordings.
    Detects R-peaks, computes RR intervals, then derives HRV metrics.

    Features: mean HR, std of RR intervals (SDNN), RMSSD, number of beats.
    """
    lead_ii = signal[:, 1]

    # Detect R-peaks: minimum height 0.3 mV, minimum distance 40 samples (400ms at 100Hz)
    peaks, _ = find_peaks(lead_ii, height=0.3, distance=40)

    features = {}

    if len(peaks) < 2:
        # Not enough peaks detected — return zeros to avoid dropping the record
        features["hr_mean"]    = 0.0
        features["rr_std"]     = 0.0
        features["rmssd"]      = 0.0
        features["beat_count"] = 0
        return features

    # RR intervals in seconds
    rr_intervals = np.diff(peaks) / SAMPLING_RATE

    # Mean heart rate in beats per minute
    features["hr_mean"]    = 60.0 / np.mean(rr_intervals)

    # SDNN: standard deviation of RR intervals — overall HRV
    features["rr_std"] = np.std(rr_intervals, ddof=1)

    # RMSSD: root mean square of successive RR differences — short-term HRV
    rr_diffs               = np.diff(rr_intervals)
    features["rmssd"] = np.sqrt(np.sum(rr_diffs ** 2) / (len(rr_diffs) - 1)) if len(rr_diffs) > 1 else 0.0
    features["beat_count"] = len(peaks)

    return features


def extract_features(filename_lr: str) -> dict | None:
    """
    Extract the full feature vector for one ECG recording.

    Returns None if the signal cannot be loaded, so the caller
    can skip the record without crashing the pipeline.
    """
    try:
        signal = load_signal(filename_lr)
        features = {}
        features.update(extract_statistical_features(signal))
        features.update(extract_hrv_features(signal))
        return features
    except Exception:
        return None


def build_feature_matrix(df: pd.DataFrame, max_records: int = None) -> pd.DataFrame:
    """
    Extract features for all records in the metadata DataFrame.

    Args:
        df:          metadata DataFrame indexed by ecg_id.
        max_records: if set, process only the first N records (useful for testing).

    Returns:
        DataFrame indexed by ecg_id with one column per feature.
    """
    rows = []
    subset = df.head(max_records) if max_records else df

    for ecg_id, row in subset.iterrows():
        features = extract_features(row["filename_lr"])
        if features is not None:
            features["ecg_id"] = ecg_id
            rows.append(features)

    result = pd.DataFrame(rows).set_index("ecg_id")
    return result


if __name__ == "__main__":
    import time
    from data_loader import load_metadata, load_scp_statements
    from labels import build_label_matrix, filter_labeled_records

    df = load_metadata()
    scp = load_scp_statements()

    # Keep only records that have at least one diagnostic superclass label
    label_matrix = build_label_matrix(df, scp)
    df_labeled, _ = filter_labeled_records(df, label_matrix)

    print(f"Extracting features for {len(df_labeled)} records...")
    print("This will take 10–20 minutes. Progress shown every 1000 records.\n")

    rows = []
    start = time.time()

    for i, (ecg_id, row) in enumerate(df_labeled.iterrows()):
        features = extract_features(row["filename_lr"])
        if features is not None:
            features["ecg_id"] = ecg_id
            rows.append(features)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start
            print(f"  {i + 1}/{len(df_labeled)} records processed ({elapsed:.0f}s elapsed)")

    feature_matrix = pd.DataFrame(rows).set_index("ecg_id")

    output_path = Path("data/features.csv")
    feature_matrix.to_csv(output_path)

    elapsed_total = time.time() - start
    print(f"\nDone. {len(feature_matrix)} records extracted in {elapsed_total:.0f}s.")
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Saved to: {output_path}")
    print(f"Failed records: {len(df_labeled) - len(feature_matrix)}")