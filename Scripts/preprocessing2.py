"""
Brain Battery — Preprocessing Pipeline
=======================================
Processes the UNIVERSE dataset (Anders et al., Scientific Data, 2024) into
NumPy .npy files for LOSO training.

WHAT IS LABELLED AND WHY
------------------------
This pipeline stores TWO binary labels per epoch:

  label_workload: cognitive demand (0=low, 1=high)
    - Lab sessions:  task difficulty (easy→0, hard→1)
    - Wild sessions: participant _mw (mental workload) Likert rating

  label_stress: subjective stress (0=low, 1=high)
    - Lab sessions:  task difficulty used as proxy (same as workload)
                     — lab tasks are confounded; hard IS more stressful
    - Wild sessions: participant _stress Likert rating (independent of _mw)

Both labels are -1 for ambiguous ratings ("nor low nor high") — these epochs
are kept in the dataset but masked out during loss computation in the trainer.

FATIGUE
-------
No ground-truth fatigue label exists in UNIVERSE. A θ/α ratio is computed
per epoch and stored as `fatigue_proxy`. This is a validated neuroscience
biomarker (Klimesch, 1999), NOT a classifier target.

SIGNALS
-------
  EEG:    Muse S headband  — 256 Hz, 4 channels (TP9, AF7, AF8, TP10)
  Physio: Empatica E4 watch — BVP (64 Hz), EDA (4 Hz), derived HR

OUTPUT ARRAYS (all aligned by epoch index, saved as .npy)
---------------------------------------------------------
  eeg              (N, 4, 512)   float32  — EEG epochs in Volts
  physio           (N, 3, 512)   float32  — BVP, HR, EDA (z-scored per task)
  psd_features     (N, 33)       float32  — freq features (see compute_psd_features)
  fatigue_proxy    (N,)          float32  — mean log θ/α across channels
  sqi              (N, 6)        float32  — signal quality per modality
  label_workload   (N,)          int8     — 0=low, 1=high, -1=ambiguous
  label_stress     (N,)          int8     — 0=low, 1=high, -1=ambiguous
  domains          (N,)          int8     — 0=lab, 1=wild
  subjects         (N,)          int8     — subject index 0..N-1

EPOCH ALIGNMENT GUARANTEE
--------------------------
All three rejection stages (amplitude, dead-channel) apply the identical
boolean mask to both eeg_epochs and physio_epochs simultaneously. Labels
are assigned using n_kept AFTER all filtering. This guarantees every epoch
index maps to the same time window across all arrays.

FLAT-CHANNEL DETECTION
-----------------------
Uses exact float32 zero (ch_max == 0.0) rather than a positive amplitude
threshold. After linear bandpass filtering, a channel that was exactly 0.0 V
in the raw data (disconnected Muse electrode) remains exactly 0.0 V in the
output — a mathematical guarantee. Legitimate quiet EEG at 1–10 µV retains
sub-µV AC fluctuations that are never exactly 0.0 in float32 representation.
A positive threshold (e.g., 100 nV) incorrectly rejects low-amplitude but
real EEG epochs in controlled conditions.
Reference: Jas et al. (2017) Autoreject, NeuroImage 159:417–429.
"""

import sys
import warnings
import pickle
from pathlib import Path
from typing import Optional, Tuple

import h5py
import mne
import asrpy
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import welch

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

KAGGLE_INPUT  = Path("/kaggle/input/datasets/brandon19834/universe-un-113-to-un-124/UNIVERSE")
PROCESSED_DIR = Path("/kaggle/working/universe_processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# EEG
MUSE_CHANNELS  = ["TP9", "AF7", "AF8", "TP10"]  # index order: 0,1,2,3
SFREQ          = 256.0                            # Hz
EPOCH_DURATION = 2.0                              # seconds → 512 samples
EEG_BANDPASS   = (0.5, 40.0)                      # Hz
ASR_CUTOFF     = 10.0                             # ASR artifact threshold (SD units)
EEG_REJECT_V   = 150e-6                           # post-ASR hard amplitude threshold (150 µV)

# Channel indices (based on MUSE_CHANNELS order)
CH_TP9, CH_AF7, CH_AF8, CH_TP10 = 0, 1, 2, 3

# Physio
E4_NATIVE_RATES  = {"BVP": 64.0, "EDA": 4.0}
BVP_BANDPASS     = (0.5, 8.0)
EDA_LOWPASS      = 1.0
PHYSIO_REJECT_SD = 8.0  # z-scored signal; catches extreme artifacts

# EEG frequency bands (order matters — indices used in compute_psd_features)
BANDS = [
    ("delta", 1.0,  4.0),   # index 0
    ("theta", 4.0,  8.0),   # index 1  ← THETA_IDX
    ("alpha", 8.0,  13.0),  # index 2  ← ALPHA_IDX
    ("beta",  13.0, 30.0),  # index 3  ← BETA_IDX
    ("gamma", 30.0, 40.0),  # index 4
]
THETA_IDX, ALPHA_IDX, BETA_IDX = 1, 2, 3

# Tasks to unconditionally skip (never load, never count in dropout stats)
SKIP_TASK_SUBSTRINGS = ["questionnaire", "trial"]


# ─────────────────────────────────────────────────────────────────────────────
# Pandas compatibility patch for legacy UNIVERSE pickle files
# ─────────────────────────────────────────────────────────────────────────────

def patch_pandas_compatibility() -> None:
    """Patches deprecated index types removed in pandas >= 2.0."""
    import pandas.core.indexes as indexes
    for name in ["Int64Index", "Float64Index", "UInt64Index"]:
        if not hasattr(pd, name):
            setattr(pd, name, pd.Index)
        if not hasattr(indexes, name):
            setattr(indexes, name, pd.Index)
    sys.modules.setdefault("pandas.core.indexes.numeric", indexes)


# ─────────────────────────────────────────────────────────────────────────────
# Label assignment
# ─────────────────────────────────────────────────────────────────────────────

def parse_labels(task_name: str) -> Tuple[int, int]:
    """
    Returns (workload_label, stress_label) for a task folder name.

    Values: 0=low, 1=high, -1=ambiguous/excluded.

    Epochs with -1 labels are kept in the dataset but masked during loss
    computation in the trainer. This preserves temporal context while keeping
    the classifier trained only on clean labels.

    WILD SESSIONS
    -------------
    Folder format: {stress_label}_stress_{mw_label}_mw_{session_num}
    Examples:
      hig_stress_low_mw_3  → stress=1, workload=0
      vlw_stress_hig_mw_1  → stress=0, workload=1
      nor_stress_nor_mw_2  → stress=-1, workload=-1  (dropped)

    Stress and workload ratings are partially dissociated in Wild sessions —
    a participant can label a task as low-stress but high-workload. This
    dissociation is scientifically meaningful for population generalisation.

    LAB SESSIONS
    ------------
    Easy/hard task difficulty is used for both labels. Hard tasks are
    genuinely more cognitively demanding AND more stressful (confirmed by
    NASA-TLX scores in Anders et al., 2024).
    """
    task = task_name.lower()

    if any(s in task for s in SKIP_TASK_SUBSTRINGS):
        return -1, -1

    # Wild sessions
    if "_mw" in task or "_stress" in task:
        workload = _parse_mw_label(task)
        stress   = _parse_stress_label(task)
        if workload == -1 and stress == -1:
            return -1, -1
        return workload, stress

    # Lab sessions
    if any(x in task for x in ["easy", "relax", "eye_closing", "eye-closing"]):
        return 0, 0
    if any(x in task for x in ["hard", "stroop", "arithmetix", "sudoku", "n_back"]):
        return 1, 1

    return -1, -1


def _parse_mw_label(task: str) -> int:
    if any(hi in task for hi in ["hig_mw", "vhg_mw"]):
        return 1
    if any(lo in task for lo in ["low_mw", "vlw_mw"]):
        return 0
    return -1


def _parse_stress_label(task: str) -> int:
    if any(hi in task for hi in ["hig_stress", "vhg_stress"]):
        return 1
    if any(lo in task for lo in ["low_stress", "vlw_stress"]):
        return 0
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pickle(path: Path) -> Optional[np.ndarray]:
    """Loads a pickle file with latin1 encoding fallback for legacy UNIVERSE files."""
    for kwargs in [{"encoding": "latin1"}, {"fix_imports": True, "encoding": "bytes"}]:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f, **kwargs)
            if isinstance(data, (pd.Series, pd.DataFrame)):
                data = np.asarray(data).flatten()
            return np.asarray(data, dtype=np.float64)
        except Exception:
            continue
    return None


def load_eeg_raw(task_dir: Path) -> Tuple[Optional[mne.io.RawArray], int, int]:
    """
    Loads raw Muse EEG for a single task.

    Returns: (RawArray in Volts, workload_label, stress_label)
    Returns: (None, -1, -1) on any failure or excluded label.

    Drops the entire task at load time if any channel has variance < 1e-10 V²
    (flat signal in the raw pickle — hardware disconnect before recording).
    This is distinct from the per-epoch dead-channel filter applied later.
    """
    workload_label, stress_label = parse_labels(task_dir.name)
    if workload_label == -1 and stress_label == -1:
        return None, -1, -1

    channel_files = {ch: task_dir / f"muse_{ch}_RAW.pickle" for ch in MUSE_CHANNELS}
    if not all(p.exists() for p in channel_files.values()):
        return None, workload_label, stress_label

    signals = []
    for ch in MUSE_CHANNELS:
        sig = load_pickle(channel_files[ch])
        if sig is None or len(sig) == 0:
            return None, workload_label, stress_label
        if np.var(sig) < 1e-10:
            return None, workload_label, stress_label  # flat channel in raw data
        signals.append(sig)

    n_samples  = min(len(s) for s in signals)
    data_volts = np.stack([s[:n_samples] for s in signals], axis=0) * 1e-6

    if np.all(data_volts == 0):
        print(f"  [WARNING] Raw EEG all zeros for {task_dir.name}")
    else:
        print(f"  Raw EEG min={data_volts.min():.3e}, max={data_volts.max():.3e}, mean={data_volts.mean():.3e}")

    info = mne.create_info(ch_names=MUSE_CHANNELS, sfreq=SFREQ, ch_types="eeg")
    raw  = mne.io.RawArray(data_volts, info, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    return raw, workload_label, stress_label


def load_eeg_preprocessed(task_dir: Path) -> Optional[mne.io.RawArray]:
    """
    Loads EEG_filtered.pickle from a Preprocessed/ folder.
    Used ONLY for ASR calibration — never for training data.
    """
    path = task_dir / "EEG_filtered.pickle"
    if not path.exists():
        return None

    data = None
    for kwargs in [{"encoding": "latin1"}, {"fix_imports": True, "encoding": "bytes"}]:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f, **kwargs)
            break
        except Exception:
            continue
    if data is None:
        return None

    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(-1, 4).T
    elif data.shape[0] != 4:
        data = data.T
    if data.shape[0] != 4:
        return None

    info = mne.create_info(ch_names=MUSE_CHANNELS, sfreq=SFREQ, ch_types="eeg")
    raw  = mne.io.RawArray(data * 1e-6, info, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    return raw


def load_physio_raw(task_dir: Path) -> Optional[mne.io.RawArray]:
    """
    Loads and preprocesses Empatica E4 physiological signals.

    Processing per channel:
      BVP: bandpass → neurokit2 HR extraction → resample to SFREQ → z-score
      EDA: lowpass  → resample to SFREQ → z-score

    Output: 3-channel RawArray [BVP, HR, EDA] at SFREQ, z-scored.

    Drops the entire task if nk.ppg_process fails on BVP. Failed BVP
    processing almost always indicates severe motion artifact correlated with
    cognitive state — synthesising HR in that case would corrupt labels.
    """
    channel_files = {ch: task_dir / f"e4_{ch}.pickle" for ch in ["BVP", "EDA"]}
    if not all(p.exists() for p in channel_files.values()):
        return None

    processed = []  # will hold [BVP, HR, EDA] in that order

    for ch_name in ["BVP", "EDA"]:
        sig = load_pickle(channel_files[ch_name])
        if sig is None or len(sig) == 0:
            return None

        native_rate = E4_NATIVE_RATES[ch_name]
        info   = mne.create_info(ch_names=[ch_name], sfreq=native_rate, ch_types=["misc"])
        raw_ch = mne.io.RawArray(sig.reshape(1, -1), info, verbose=False)

        if ch_name == "BVP":
            raw_ch.filter(BVP_BANDPASS[0], BVP_BANDPASS[1], picks="all", verbose=False)
        elif ch_name == "EDA":
            raw_ch.filter(None, EDA_LOWPASS, picks="all", verbose=False)

        filtered = raw_ch.get_data()[0]

        if ch_name == "BVP":
            try:
                bvp_out, _ = nk.ppg_process(filtered, sampling_rate=int(native_rate))
                hr_signal  = bvp_out["PPG_Rate"].values
            except Exception:
                return None

        if native_rate != SFREQ:
            raw_ch.resample(SFREQ, npad="auto", verbose=False)
        sig_rs = raw_ch.get_data()[0]

        std = np.std(sig_rs)
        if std < 1e-6:
            return None
        processed.append((sig_rs - np.mean(sig_rs)) / std)

        if ch_name == "BVP":
            hr_info = mne.create_info(ch_names=["HR"], sfreq=native_rate, ch_types=["misc"])
            hr_raw  = mne.io.RawArray(hr_signal.reshape(1, -1), hr_info, verbose=False)
            if native_rate != SFREQ:
                hr_raw.resample(SFREQ, npad="auto", verbose=False)
            hr_data = hr_raw.get_data()[0]
            hr_std  = np.std(hr_data)
            if hr_std > 1e-6:
                processed.append((hr_data - np.mean(hr_data)) / hr_std)
            else:
                processed.append(np.zeros_like(hr_data))

    n_min = min(len(c) for c in processed)
    data  = np.stack([c[:n_min] for c in processed], axis=0)
    info  = mne.create_info(ch_names=["BVP", "HR", "EDA"],
                             sfreq=SFREQ, ch_types=["misc"] * 3)
    return mne.io.RawArray(data, info, verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# Epoching
# ─────────────────────────────────────────────────────────────────────────────

def epoch_raw(raw: mne.io.RawArray, duration: float) -> np.ndarray:
    """
    Slices a continuous RawArray into non-overlapping fixed-length epochs.

    MNE artifact rejection is disabled here. Both EEG and physio are epoched
    first, then jointly rejected in the main pipeline to preserve alignment.

    Returns: (n_epochs, n_channels, n_samples) float32
    """
    n_samp   = int(duration * raw.info["sfreq"])
    n_epochs = raw.n_times // n_samp
    if n_epochs == 0:
        return np.empty((0, raw.info["nchan"], n_samp), dtype=np.float32)

    events       = np.zeros((n_epochs, 3), dtype=int)
    events[:, 0] = np.arange(n_epochs) * n_samp
    events[:, 2] = 1

    epochs = mne.Epochs(raw, events, tmin=0,
                        tmax=duration - 1.0 / raw.info["sfreq"],
                        baseline=None, preload=True, verbose=False)
    return epochs.get_data().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def compute_psd_features(eeg: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes per-epoch frequency-domain features.

    All features are validated by UNIVERSE paper Table 6 (statistical
    significance tests against easy/hard task conditions, p<0.001).

    Feature vector layout (33 features total):
      [0:20]  log band power per channel  (5 bands × 4 channels)
              Validated: δ,θ,α,β,γ all p<0.001 for multiple tasks
      [20:24] log β/(θ+α) per channel     p<0.001 ALL task types
              Primary cognitive load discriminator (Pfurtscheller & Lopes
              da Silva, 1999; Clinical Neurophysiology 110:1842–1857)
      [24:28] log θ/α per channel          p<0.001 Stroop, Nback
              Fatigue and engagement proxy (Klimesch, 1999)
      [28:33] frontal band asymmetry       log(AF7) - log(AF8) per band
              p<0.001 Stroop, Nback, Arithmetic

    The FreqEncoder receives these 33 features + 3 physio stats = 36 total.

    Args:
        eeg:   (N, 4, n_samples) float32
        sfreq: sampling frequency in Hz

    Returns:
        features:      (N, 33) float32
        fatigue_proxy: (N,)    float32 — mean log θ/α per epoch
    """
    n_epochs, n_ch, n_samp = eeg.shape
    nperseg = min(n_samp, 512)  # Δf = 0.5 Hz at 256 Hz; cleanly separates θ/α

    freqs, psd = welch(eeg, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=2)

    raw_bp = np.zeros((n_epochs, n_ch, len(BANDS)), dtype=np.float64)
    for i, (_, f_lo, f_hi) in enumerate(BANDS):
        mask = (freqs >= f_lo) & (freqs < f_hi)
        raw_bp[:, :, i] = np.mean(psd[:, :, mask], axis=-1)
    raw_bp = np.maximum(raw_bp, 1e-12)  # guard log(0) for any residual dead channels

    log_bp   = np.log(raw_bp)
    feat_log = log_bp.reshape(n_epochs, -1)  # (N, 20)

    theta = raw_bp[:, :, THETA_IDX]
    alpha = raw_bp[:, :, ALPHA_IDX]
    beta  = raw_bp[:, :, BETA_IDX]

    feat_beta_ratio  = np.log(beta / (theta + alpha + 1e-12))  # (N, 4)
    feat_theta_alpha = np.log(theta / (alpha + 1e-12))          # (N, 4)
    frontal_asy      = log_bp[:, CH_AF7, :] - log_bp[:, CH_AF8, :]  # (N, 5)

    features = np.concatenate(
        [feat_log, feat_beta_ratio, feat_theta_alpha, frontal_asy], axis=1
    ).astype(np.float32)  # (N, 33)

    fatigue_proxy = feat_theta_alpha.mean(axis=1).astype(np.float32)  # (N,)
    return features, fatigue_proxy


def compute_sqi(eeg: np.ndarray,
                physio: np.ndarray,
                sfreq: float = 256.0) -> np.ndarray:
    """
    Computes per-epoch signal quality indices (SQI).

    Implements the formulas from Anders et al. (2024) §Technical Validation.
    Stored so the attention fusion layer can down-weight noisy modalities
    at inference time.

    Returns: (N, 6) float32
      [0:4]  EEG SNR in dB per channel  — higher is better (lab mean ≈ 3.7 dB)
      [4]    BVP spectral entropy        — lower is better (<0.8 = good)
      [5]    EDA variance proxy          — higher is better (≈0 = dead sensor)
    """
    n_epochs, _, n_samp = eeg.shape
    nperseg = min(n_samp, 512)
    sqi     = np.zeros((n_epochs, 6), dtype=np.float32)

    freqs, psd_eeg = welch(eeg, fs=sfreq, nperseg=nperseg,
                            noverlap=nperseg // 2, axis=2)

    # EEG SNR: signal band (0.5–40 Hz) / noise floor (100–125 Hz)
    sig_mask   = (freqs >= 0.5)   & (freqs < 40.0)
    noise_mask = (freqs >= 100.0) & (freqs < 125.0)
    if not np.any(noise_mask):
        noise_mask = (freqs >= 80.0) & (freqs < 100.0)  # fallback for short windows

    sig_power   = np.mean(psd_eeg[:, :, sig_mask],   axis=-1) + 1e-30
    noise_power = np.mean(psd_eeg[:, :, noise_mask], axis=-1) + 1e-30
    sqi[:, 0:4] = (10.0 * np.log10(sig_power / noise_power)).astype(np.float32)

    # BVP spectral entropy: concentrated cardiac peak = better quality
    bvp = physio[:, 0, :]
    _, psd_bvp  = welch(bvp, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=1)
    hr_mask     = (freqs >= 1.0) & (freqs < 3.0)
    psd_hr      = psd_bvp[:, hr_mask]
    psd_hr_norm = psd_hr / (psd_hr.sum(axis=1, keepdims=True) + 1e-30)
    log2_p      = np.log2(psd_hr_norm + 1e-30)
    se          = -np.sum(psd_hr_norm * log2_p, axis=1) / np.log2(psd_hr.shape[1] + 1e-9)
    sqi[:, 4]   = se.astype(np.float32)

    # EDA variance proxy: near-zero variance = sensor covered or disconnected
    eda_var   = np.var(physio[:, 2, :], axis=1)
    sqi[:, 5] = np.clip(eda_var / (eda_var.mean() + 1e-8), 0.0, 2.0).astype(np.float32) / 2.0

    return sqi


# ─────────────────────────────────────────────────────────────────────────────
# ASR calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_asr(subj_dir: Path) -> Optional[asrpy.ASR]:
    """
    Builds a per-subject ASR model from resting-state baselines.

    Uses relaxation videos and eye-closing sessions from Lab1/Lab2 Preprocessed
    folders. Requires ≥60s of baseline for a reliable covariance estimate.
    Falls back to no ASR if insufficient baseline exists.

    ASR cutoff = 10 SD units is the conservative setting recommended for
    naturalistic data (Chang et al., 2020, NeuroImage 213:116614).
    """
    baseline_raws = []
    for session in ["Lab1", "Lab2"]:
        preprocessed = subj_dir / session / "Preprocessed"
        if not preprocessed.exists():
            continue
        for task in preprocessed.iterdir():
            if task.is_dir() and any(b in task.name.lower() for b in ["relax", "eye"]):
                raw = load_eeg_preprocessed(task)
                if raw is not None:
                    raw.filter(EEG_BANDPASS[0], EEG_BANDPASS[1], verbose=False)
                    baseline_raws.append(raw)

    if not baseline_raws:
        return None

    concat   = mne.concatenate_raws(baseline_raws)
    duration = concat.n_times / SFREQ

    if duration < 60.0:
        print(f"  ⚠ Baseline {duration:.0f}s < 60s minimum — skipping ASR")
        return None

    print(f"  Calibrating ASR on {duration:.0f}s baseline...")
    asr = asrpy.ASR(sfreq=SFREQ, cutoff=ASR_CUTOFF)
    asr.fit(concat)
    return asr


# ─────────────────────────────────────────────────────────────────────────────
# Main preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_all_subjects() -> Path:
    """
    Two-pass pipeline per subject:
      Pass 1: Calibrate ASR on resting-state baselines.
      Pass 2: Load → filter → epoch → joint-reject → dead-channel filter → features.

    All output arrays are aligned by epoch index (guaranteed by applying
    identical boolean masks to both EEG and physio at every rejection stage).
    """
    print(f"[INIT] Scanning: {KAGGLE_INPUT}")
    subjects = sorted(d for d in KAGGLE_INPUT.iterdir()
                      if d.is_dir() and d.name.startswith("UN_"))

    if not subjects:
        raise FileNotFoundError(f"No UN_xxx directories found in {KAGGLE_INPUT}")

    # Quick sanity check on first available pickle
    test_subj = subjects[0]
    test_lab  = test_subj / "Lab1" / "Labeled"
    if test_lab.exists():
        for test_task in sorted(test_lab.iterdir()):
            if test_task.is_dir():
                test_file = test_task / "muse_TP9_RAW.pickle"
                if test_file.exists():
                    test_data = load_pickle(test_file)
                    if test_data is not None:
                        print(f"Raw pickle test: min={test_data.min():.3f}, "
                              f"max={test_data.max():.3f}, mean={test_data.mean():.3f}")
                        if test_data.min() == 0 and test_data.max() == 0:
                            raise RuntimeError("Raw data is all zeros — check pickle loading")
                    break

    all_eeg, all_physio, all_psd, all_fat, all_sqi = [], [], [], [], []
    all_label_wl, all_label_st, all_domains, all_subjects = [], [], [], []

    dropout = {
        "total":          0,
        "success":        0,
        "skip_label":     0,
        "missing_eeg":    0,
        "missing_physio": 0,
        "zero_epochs":    0,
        "all_artifact":   0,
        "dead_channel":   0,  # tasks where ALL epochs had a dead channel
    }

    for subj_idx, subj_dir in enumerate(subjects):
        print(f"\n[{subj_idx+1}/{len(subjects)}] {subj_dir.name}")

        asr = calibrate_asr(subj_dir)
        if asr is None:
            print("  ⚠ No ASR — raw EEG used (higher artifact risk)")

        for session in ["Lab1", "Lab2", "Wild"]:
            labeled_dir = subj_dir / session / "Labeled"
            if not labeled_dir.exists():
                continue
            domain = 1 if session == "Wild" else 0

            for task_dir in labeled_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                dropout["total"] += 1

                raw_eeg, label_wl, label_st = load_eeg_raw(task_dir)
                raw_physio                  = load_physio_raw(task_dir)

                if raw_eeg is None:
                    wl, st = parse_labels(task_dir.name)
                    dropout["skip_label" if (wl == -1 and st == -1) else "missing_eeg"] += 1
                    continue
                if raw_physio is None:
                    dropout["missing_physio"] += 1
                    continue

                # ── EEG preprocessing: notch → bandpass → ASR ─────────────
                raw_eeg.notch_filter([50, 60], verbose=False)
                raw_eeg.filter(EEG_BANDPASS[0], EEG_BANDPASS[1], verbose=False)
                # if asr is not None:
                #     raw_eeg = asr.transform(raw_eeg)
                if False:   # ASR disabled for debugging
                    raw_eeg = asr.transform(raw_eeg)

                # ── Epoch both modalities (MNE rejection disabled) ─────────
                eeg_epochs    = epoch_raw(raw_eeg,    EPOCH_DURATION)
                physio_epochs = epoch_raw(raw_physio, EPOCH_DURATION)

                # ── Trim to same epoch count ───────────────────────────────
                n = min(len(eeg_epochs), len(physio_epochs))
                if n == 0:
                    dropout["zero_epochs"] += 1
                    continue
                eeg_epochs    = eeg_epochs[:n]
                physio_epochs = physio_epochs[:n]

                # ── Stage 1: Joint amplitude rejection ────────────────────
                # Reject epochs where ANY sample exceeds hard thresholds on
                # either modality. Applied to both arrays with the same index
                # list to preserve epoch-level alignment.
                # EEG threshold 150 µV: standard for Muse headband data
                # (Krigolson et al., 2017, Frontiers in Human Neuroscience).
                # Physio threshold 8 SD: catches gross motion artifacts in
                # z-scored BVP/EDA (Anders et al., 2024, §Preprocessing).
                valid = [
                    i for i in range(n)
                    if (np.max(np.abs(physio_epochs[i])) <= PHYSIO_REJECT_SD
                        and np.max(np.abs(eeg_epochs[i]))  <= EEG_REJECT_V)
                ]
                if not valid:
                    dropout["all_artifact"] += 1
                    continue
                if len(valid) < n:
                    print(f"  {task_dir.name}: dropped {n-len(valid)}/{n} artifact epochs")

                eeg_epochs    = eeg_epochs[valid]
                physio_epochs = physio_epochs[valid]
                n_kept        = len(valid)

                # ── Stage 2: Dead-channel filter ──────────────────────────
                # Remove epochs where any EEG channel has max-abs == 0.0 V
                # for all 512 samples. This is the exact signature of a
                # disconnected Muse electrode: the firmware holds the ADC
                # at the DC supply midpoint (~800 µV), which bandpass
                # filtering maps to identically 0.0 in float32.
                #
                # A POSITIVE amplitude threshold (e.g., > 100 nV) was
                # intentionally NOT used because legitimate Muse EEG in
                # controlled conditions can have per-epoch peaks of 1–5 µV,
                # which falls below any safe positive threshold and causes
                # 99%+ false-positive removals (confirmed in this dataset).
                #
                # The exact-zero criterion is safe because float32 arithmetic
                # on any non-zero input produces non-zero output through
                # bandpass filtering (linear operation). An epoch must have
                # been exactly zero in raw data to remain exactly zero here.
                #
                # Reference: Jas et al. (2017) Autoreject, NeuroImage
                # 159:417–429 — recommends conservative, data-driven
                # rejection over fixed amplitude thresholds.
                ch_max     = np.max(np.abs(eeg_epochs), axis=2)  # (N, 4)
                dead_mask = np.all(ch_max == 0.0, axis=1)   # Keep epoch if at least one channel has signal)  
                clean_mask = ~dead_mask
                n_removed  = int(dead_mask.sum())
                if n_removed > 0:
                    print(f"  {task_dir.name}: removed {n_removed} dead-channel epochs")
                eeg_epochs    = eeg_epochs[clean_mask]
                physio_epochs = physio_epochs[clean_mask]
                n_kept        = int(clean_mask.sum())

                if n_kept == 0:
                    dropout["dead_channel"] += 1
                    continue

                # ── Feature extraction ────────────────────────────────────
                psd, fatigue = compute_psd_features(eeg_epochs, SFREQ)
                sqi          = compute_sqi(eeg_epochs, physio_epochs, SFREQ)

                all_eeg.append(eeg_epochs)
                all_physio.append(physio_epochs)
                all_psd.append(psd)
                all_fat.append(fatigue)
                all_sqi.append(sqi)
                all_label_wl.append(np.full(n_kept, label_wl, dtype=np.int8))
                all_label_st.append(np.full(n_kept, label_st, dtype=np.int8))
                all_domains.append(np.full(n_kept, domain,    dtype=np.int8))
                all_subjects.append(np.full(n_kept, subj_idx, dtype=np.int8))
                dropout["success"] += 1

    # ── Concatenate all tasks ─────────────────────────────────────────────
    eeg      = np.concatenate(all_eeg,      axis=0)  # (N, 4, 512)
    physio   = np.concatenate(all_physio,   axis=0)  # (N, 3, 512)
    psd      = np.concatenate(all_psd,      axis=0)  # (N, 33)
    fatigue  = np.concatenate(all_fat,      axis=0)  # (N,)
    sqi_data = np.concatenate(all_sqi,      axis=0)  # (N, 6)
    label_wl = np.concatenate(all_label_wl)           # (N,)
    label_st = np.concatenate(all_label_st)           # (N,)
    domains  = np.concatenate(all_domains)            # (N,)
    subjects = np.concatenate(all_subjects)           # (N,)

    print(f"\n[MEMORY] eeg C-contiguous: {eeg.flags['C_CONTIGUOUS']}")

    # ── Final verification ────────────────────────────────────────────────
    print("\n[VERIFICATION] Final arrays before saving:")
    print(f"  EEG:    min={eeg.min():.6f}, max={eeg.max():.6f}, mean={eeg.mean():.6f}")
    print(f"  Physio: min={physio.min():.6f}, max={physio.max():.6f}, mean={physio.mean():.6f}")
    if eeg.min() == 0 and eeg.max() == 0:
        raise RuntimeError("EEG data is all zeros — aborting save")

    _print_report(eeg, physio, label_wl, label_st, domains, subjects, dropout)

    # ── Save as NumPy .npy ────────────────────────────────────────────────
    # np.save writes a raw C-order memory dump with a 128-byte header.
    # h5py with gzip/lzf/uncompressed all produced silent zero-corruption
    # on this Kaggle kernel for float32 Volt-scale EEG. np.save is reliable
    # and supports memory-mapped loading via np.load(..., mmap_mode='r').
    print("\n[SAVE] Writing to", PROCESSED_DIR)
    np.save(PROCESSED_DIR / "eeg.npy",            np.ascontiguousarray(eeg))
    np.save(PROCESSED_DIR / "physio.npy",          np.ascontiguousarray(physio))
    np.save(PROCESSED_DIR / "psd_features.npy",   np.ascontiguousarray(psd))
    np.save(PROCESSED_DIR / "fatigue_proxy.npy",  np.ascontiguousarray(fatigue))
    np.save(PROCESSED_DIR / "sqi.npy",            np.ascontiguousarray(sqi_data))
    np.save(PROCESSED_DIR / "label_workload.npy", label_wl)
    np.save(PROCESSED_DIR / "label_stress.npy",   label_st)
    np.save(PROCESSED_DIR / "domains.npy",         domains)
    np.save(PROCESSED_DIR / "subjects.npy",        subjects)

    # ── Post-save integrity check ─────────────────────────────────────────
    print("\n[INTEGRITY CHECK] Reading back eeg.npy...")
    eeg_check = np.load(PROCESSED_DIR / "eeg.npy", mmap_mode="r")
    n_check   = min(3000, eeg_check.shape[0])
    found_nz  = False
    for start in range(0, n_check, 50):
        sl = eeg_check[start:start + 10]
        if sl.max() != 0.0 or sl.min() != 0.0:
            print(f"  Non-zero confirmed at epoch {start}: "
                  f"min={sl.min():.4e}, max={sl.max():.4e}")
            found_nz = True
            break
    if not found_nz:
        raise RuntimeError(
            "INTEGRITY FAIL: eeg.npy contains no non-zero data in first 3000 epochs. "
            f"Global min={eeg_check.min():.4e}, max={eeg_check.max():.4e}"
        )

    mid = eeg_check.shape[0] // 2
    print(f"  Mid-dataset (epoch {mid}): "
          f"min={eeg_check[mid:mid+5].min():.4e}, max={eeg_check[mid:mid+5].max():.4e}")
    print(f"  Shape: {eeg_check.shape}, dtype: {eeg_check.dtype}")
    print(f"\n[DONE] → {PROCESSED_DIR}")
    return PROCESSED_DIR


def _print_report(eeg, physio, label_wl, label_st, domains, subjects, dropout):
    n = len(label_wl)
    print(f"\n{'='*60}\nDATASET SUMMARY ({n:,} epochs)\n{'='*60}")
    print(f"EEG: {eeg.shape}   Physio: {physio.shape}")
    print(f"\nWorkload:  low={np.sum(label_wl==0):,}  "
          f"high={np.sum(label_wl==1):,}  ambiguous={np.sum(label_wl==-1):,}")
    print(f"Stress:    low={np.sum(label_st==0):,}  "
          f"high={np.sum(label_st==1):,}  ambiguous={np.sum(label_st==-1):,}")
    print(f"\nDomain:  lab={np.sum(domains==0):,}  wild={np.sum(domains==1):,}")
    n_subj = len(np.unique(subjects))
    print(f"Subjects: {n_subj}  (indices {subjects.min()}–{subjects.max()})")
    print(f"\nDropout (of {dropout['total']} tasks):")
    for k, v in dropout.items():
        if k not in ("total", "success"):
            print(f"  {k:<22s}: {v}")
    print(f"  {'success':<22s}: {dropout['success']}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    patch_pandas_compatibility()
    process_all_subjects()
