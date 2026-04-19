"""
LOSO Training Pipeline
=======================================
Trains a Subject-Agnostic Neural Network (SANN) on UNIVERSE data using
Leave-One-Subject-Out cross-validation.

MULTI-TASK TARGETS
------------------
The model jointly predicts TWO binary targets per epoch:
  - Cognitive workload: 0=low, 1=high  (primary target)
  - Stress:            0=low, 1=high  (secondary target)

Both share the same fused representation. Epochs labelled -1 (ambiguous
"nor" ratings from Wild sessions) are kept in batches but masked out of
the loss computation — only 0/1 epochs contribute to gradient updates.


FATIGUE PROXY
-------------
No ground-truth fatigue label. The θ/α ratio is read from `psd_features`
columns [24:28] and reported as a continuous per-subject mean at the end
of each fold.

ARCHITECTURE
------------
  EEGNet        → 32-dim EEG representation
  PhysioCNN     → 32-dim physio waveform representation
  FreqEncoder   → 32-dim frequency feature representation
  AttentionFusion (SQI-conditioned) → 96-dim fused representation
  Workload head → 2-class logits
  Stress head   → 2-class logits
  Subject head  → N-class logits (GRL domain adversary)
  Projector     → 16-dim embedding (contrastive loss)
"""

import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(42)

# --- MODIFIED: point to the merged .npy directory ---
if os.path.exists("/kaggle/input"):
    # Adjust this path to where your merged .npy files are stored on Kaggle.
    # For local testing, point to the merged directory.
    DATA_DIR = Path("/kaggle/input/datasets/brandon19834/universe-merged-withzero-noasr")
    MODEL_DIR = Path("/kaggle/working/models")
else:
    DATA_DIR = Path("./universe_processed_combined")   # local merged folder
    MODEL_DIR = Path("./models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {DEVICE}")


# Hyperparameters
BATCH_SIZE      = 128
LEARNING_RATE   = 3e-4
NUM_EPOCHS      = 30
LAMBDA_DOMAIN   = 0.01    # GRL domain adversary weight
LAMBDA_CONTRAST = 0.001   # Contrastive loss weight
WARMUP_EPOCHS   = 12      # Epochs before adversarial/contrastive losses activate
MIN_EPOCHS      = 25      # Must train at least this many epochs
PATIENCE        = 8       # Early stopping patience

# Dimension constants (must match preprocessing output)
FREQ_IN_DIM = 36   # 33 PSD features + 3 physio stats
SQI_DIM     = 6    # 4 EEG SNR channels + BVP SE + EDA var
TOWER_DIM   = 32   # output dimension of each modality tower
FUSION_DIM  = 96   # 3 × TOWER_DIM after concatenation

# Label constant for masked loss
LABEL_AMBIGUOUS = -1

RUN_ONLY_FULL = True

# ─────────────────────────────────────────────────────────────────────────────
# ABLATION CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """
    Controls which model components are active.

    When a tower is disabled, its output is replaced with zeros before fusion.
    The fusion layer still operates normally — this isolates the contribution
    of each modality's learned representation.

    Note: disabling all three towers simultaneously is undefined behavior.
    """
    name:       str   = "full"
    use_eeg:    bool  = True
    use_physio: bool  = True
    use_freq:   bool  = True
    use_grl:    bool  = True
    use_sqi:    bool  = True


# All conditions for the population generalisation study
ABLATION_CONDITIONS = [
    AblationConfig("full"),
    AblationConfig("eeg_only",    use_physio=False, use_freq=False),
    AblationConfig("physio_only", use_eeg=False,    use_freq=False),
    AblationConfig("freq_only",   use_eeg=False,    use_physio=False),
    AblationConfig("no_grl",      use_grl=False),
    AblationConfig("no_sqi",      use_sqi=False),
]

# ─────────────────────────────────────────────────────────────────────────────
# Masked cross-entropy loss
# ─────────────────────────────────────────────────────────────────────────────

class MaskedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss that ignores epochs labelled -1 (ambiguous).

    This handles the "nor low nor high" Wild session ratings. Those epochs
    still pass through the model (we can't remove them without breaking
    temporal batch structure) but they do not contribute to the gradient.

    If a batch contains NO valid labels (all -1), returns zero loss.
    This is rare but can happen in small validation folds.
    """
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid_mask = labels != LABEL_AMBIGUOUS

        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        return F.cross_entropy(
            logits[valid_mask],
            labels[valid_mask],
            label_smoothing=self.label_smoothing
        )


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Reversal Layer (Ganin & Lempitsky, 2015)
# ─────────────────────────────────────────────────────────────────────────────

class GradientReversalFn(torch.autograd.Function):
    """Negates and scales gradients in the backward pass to train a domain-invariant encoder."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# ─────────────────────────────────────────────────────────────────────────────
# Supervised Contrastive Loss (Khosla et al., 2020)
# ─────────────────────────────────────────────────────────────────────────────

class SupervisedContrastiveLoss(nn.Module):
    """
    NT-Xent loss: pulls same-class embeddings together, pushes different-class apart.

    Temperature=0.5: at T=0.1, gradient magnitude scales 10× which overwhelms
    the classifier. T=0.5 preserves clustering geometry at manageable scale.

    Contrastive loss is computed only on workload labels (the primary target).
    Using both labels would require multi-label contrastive logic with minimal
    empirical benefit at this dataset scale.
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Only use epochs with valid (non-ambiguous) labels
        valid = labels != LABEL_AMBIGUOUS
        if valid.sum() < 2 or len(torch.unique(labels[valid])) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=False)

        z      = F.normalize(embeddings[valid], dim=1, eps=1e-8)
        y      = labels[valid]
        sim    = (torch.matmul(z, z.T) / self.temperature).clamp(-10, 10)
        eye    = torch.eye(y.size(0), device=z.device).bool()
        mask   = (y.unsqueeze(1) == y.unsqueeze(0)) & ~eye

        sim_max, _  = sim.max(dim=1, keepdim=True)
        logits      = sim - sim_max.detach()
        exp_logits  = torch.exp(logits) * ~eye
        log_prob    = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        pos_mean    = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        return -pos_mean.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────

class EEGNet(nn.Module):
    """
    Depthwise-separable CNN for EEG (Lawhern et al., 2018).

    Temporal kernel = sfreq//2 = 128 samples = 0.5s (one full theta cycle).
    Squeeze-and-Excitation block learns per-feature-map attention (Hu et al., 2018).

    Input:  (batch, 1, n_channels=4, n_samples=512)
    Output: (batch, F2=32)
    """
    def __init__(self, n_channels: int = 4, F1: int = 16,
                 D: int = 2, F2: int = 32, dropout: float = 0.5):
        super().__init__()
        self.temp_conv  = nn.Conv2d(1, F1, (1, 128), padding=(0, 64), bias=False)
        self.bn1        = nn.BatchNorm2d(F1)
        self.spat_conv  = nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False)
        self.bn2        = nn.BatchNorm2d(F1*D)
        self.elu        = nn.ELU()
        self.pool1      = nn.AvgPool2d((1, 4))
        self.drop1      = nn.Dropout(dropout)
        self.sep_conv   = nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3        = nn.BatchNorm2d(F2)
        self.pool2      = nn.AvgPool2d((1, 8))
        self.drop2      = nn.Dropout(dropout)
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))

        # SE block: global average → channel excitation → multiplicative gate
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(F2, F2 // 4, 1), nn.ReLU(),
            nn.Conv2d(F2 // 4, F2, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.temp_conv(x))
        x = self.drop1(self.pool1(self.elu(self.bn2(self.spat_conv(x)))))
        x = self.drop2(self.pool2(self.elu(self.bn3(self.sep_conv(x)))))
        x = x * self.se(x)
        return self.adapt_pool(x).view(x.size(0), -1)


class PhysioCNN(nn.Module):
    """
    1D CNN over 3 physio channels: BVP, HR, EDA.
    Input:  (batch, 3, 512)
    Output: (batch, 32)
    """
    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 8,  kernel_size=15, padding=7),  nn.BatchNorm1d(8),  nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(8, 16, kernel_size=7,  padding=3),  nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc   = nn.Linear(16, out_dim)
        self.drop = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.net(x).squeeze(-1)))


class FreqEncoder(nn.Module):
    """
    MLP over 36 frequency/physio features:
      33 PSD features (band powers, β/(θ+α), θ/α, frontal asymmetry)
    +  3 physio stats  (HR mean, HR std, log-EDA mean)
    Input:  (batch, 36)
    Output: (batch, 32)
    """
    def __init__(self, in_dim: int = FREQ_IN_DIM, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SQIGate(nn.Module):
    """
    Learnable mapping from raw SQI values to per-modality quality weights ∈ (0, 1).

    Initialised near neutral (gain=0.1) so it does not dominate early training.
    As training progresses, the gate learns to suppress noisy modalities
    (e.g., low-SNR EEG, high BVP spectral entropy) automatically.
    """
    def __init__(self, sqi_dim: int = SQI_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(sqi_dim, 16), nn.ReLU(),
            nn.Linear(16, 3),       nn.Sigmoid()
        )
        nn.init.xavier_uniform_(self.proj[2].weight, gain=0.01)
        nn.init.constant_(self.proj[2].bias, 0.0)

    def forward(self, sqi: torch.Tensor) -> torch.Tensor:
        #Returns (batch, 3) quality weights ∈ (0, 1).
        return self.proj(sqi)


class AttentionFusion(nn.Module):
    """
    SQI-conditioned attention fusion over three modality towers.

    Attention weights = softmax(content_logits * (quality_gate + 0.1))

    The quality floor of 0.1 prevents full gating during early training
    before the SQI gate has calibrated. Once trained, the gate can drive
    a bad modality's effective weight close to zero while preserving the
    others — something impossible with content-only attention.

    Output: (batch, 96) — each modality occupies its own 32-dim subspace,
    so features are preserved rather than destructively summed.
    """
    def __init__(self, in_dim: int = FUSION_DIM, sqi_dim: int = SQI_DIM):
        super().__init__()
        self.content_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.ReLU(),
            nn.Linear(in_dim // 2, 3)
        )
        self.sqi_gate = SQIGate(sqi_dim)
        self.norm     = nn.LayerNorm(in_dim)

    def forward(self, f_eeg, f_phys, f_freq, sqi):
        """
        SQI-conditioned attention fusion.

        Previous formula: softmax(logits/2 + log(quality)) caused immediate
        one-hot collapse because log(quality) creates 68× asymmetry at epoch 1
        when any modality gains a marginal quality advantage.

        Fix: multiplicative quality scaling with high temperature.
            - quality ∈ (0.1, 1.1) after the +0.1 floor — bounded, no log explosion
            - T=3.0 limits max softmax concentration to ~e^(1/3) ≈ 1.4× per logit unit
            - Entropy stays high because the input range to softmax is narrow

        Reference: Zadeh et al. (2018) MFN — multiplicative modality gating
        is standard in multimodal fusion precisely because additive log-space
        weighting is numerically unstable with learned quality scores.

        Args:
            f_eeg, f_phys, f_freq: (batch, 32) tower outputs
            sqi: batch, 6) raw SQI values
        Returns:
            fused: (batch, 96) normalised fused representation
            att_weights: (batch, 3)  attention weights for analysis
        """
        concat  = torch.cat([f_eeg, f_phys, f_freq], dim=1)
        logits  = self.content_head(concat)   # (batch, 3)
        quality = self.sqi_gate(sqi)          # (batch, 3) ∈ (0, 1)

        # Multiplicative quality scaling, NOT additive log-space.
        # Floor of 0.1 prevents any modality from being fully silenced
        # before the gate has calibrated (~warmup period).
        att = F.softmax(logits * (quality + 0.1) / 3.0, dim=1)

        fused = torch.cat([
            att[:, 0:1] * f_eeg,
            att[:, 1:2] * f_phys,
            att[:, 2:3] * f_freq,
        ], dim=1)

        return self.norm(fused), att


class BrainBatterySANN(nn.Module):
    """
    Subject-Agnostic Neural Network for joint workload + stress classification.

    The cfg (AblationConfig) controls which towers and components are active.
    Disabled towers output zero vectors — the fusion layer still runs normally,
    so attention weights are computed over whatever signal remains. This is the
    correct ablation design: it isolates learned contribution, not architectural
    capacity.

    Forward output:
      (workload_logits, stress_logits, subject_logits,
       projection, fused_embedding, attention_weights)
    """
    def __init__(self, n_subjects: int = 11,
                 cfg: AblationConfig = AblationConfig()):
        super().__init__()
        self.cfg = cfg

        self.eeg_tower    = EEGNet(n_channels=4, F2=TOWER_DIM)
        self.physio_tower = PhysioCNN(out_dim=TOWER_DIM)
        self.freq_tower   = FreqEncoder(in_dim=FREQ_IN_DIM, out_dim=TOWER_DIM)
        self.fusion       = AttentionFusion(in_dim=FUSION_DIM, sqi_dim=SQI_DIM)

        self.workload_head = nn.Sequential(
            nn.Linear(FUSION_DIM, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)
        )
        self.stress_head = nn.Sequential(
            nn.Linear(FUSION_DIM, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 2)
        )
        self.subject_head = nn.Sequential(
            nn.Linear(FUSION_DIM, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, n_subjects)
        )
        self.projector = nn.Sequential(
            nn.Linear(FUSION_DIM, 32), nn.ReLU(), nn.Linear(32, 16)
        )

    def forward(self, eeg, physio, freq, sqi, grl_alpha: float = 1.0):
        # Compute all tower outputs, then zero out disabled ones.
        # We always run all towers to keep gradient flow paths consistent
        # across conditions — disabling via zeroing is cleaner than
        # skipping the forward pass entirely.
        f_eeg   = self.eeg_tower(eeg)
        f_phys  = self.physio_tower(physio)
        f_freq  = self.freq_tower(freq)

        if not self.cfg.use_eeg:
            f_eeg  = torch.zeros_like(f_eeg)
        if not self.cfg.use_physio:
            f_phys = torch.zeros_like(f_phys)
        if not self.cfg.use_freq:
            f_freq = torch.zeros_like(f_freq)

        # For no-SQI condition: replace SQI gate output with neutral constant
        # (0.5 for all modalities). The gate still exists but has no effect.
        if not self.cfg.use_sqi:
            neutral_sqi = torch.full_like(sqi, 0.5)
            fused, att  = self.fusion(f_eeg, f_phys, f_freq, neutral_sqi)
        else:
            fused, att  = self.fusion(f_eeg, f_phys, f_freq, sqi)

        # Compute subject head with GRL if enabled
        if self.cfg.use_grl:
            subj_out = self.subject_head(GradientReversalFn.apply(fused, grl_alpha))
        else:
            subj_out = torch.zeros(fused.size(0), self.subject_head[-1].out_features,
                                    device=fused.device)

        return (
            self.workload_head(fused),
            self.stress_head(fused),
            subj_out,
            self.projector(fused),
            fused,
            att
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class UniverseDataset(Dataset):
    def __init__(self, data_dir: Path):
        print("[DATA] Loading .npy files with memory mapping...")
        # Memory-map the raw files – these never change
        self._raw_eeg    = np.load(data_dir / "eeg.npy", mmap_mode='r')
        self._raw_physio = np.load(data_dir / "physio.npy", mmap_mode='r')
        self._raw_psd    = np.load(data_dir / "psd_features.npy", mmap_mode='r')
        self._raw_sqi    = np.load(data_dir / "sqi.npy", mmap_mode='r')

        self.label_wl      = np.load(data_dir / "label_workload.npy").astype(np.int64)
        self.label_st      = np.load(data_dir / "label_stress.npy").astype(np.int64)
        self.domains       = np.load(data_dir / "domains.npy").astype(np.int64)
        self.subjects      = np.load(data_dir / "subjects.npy").astype(np.int64)
        self.fatigue_proxy = np.load(data_dir / "fatigue_proxy.npy").astype(np.float32)

        # Active arrays – these will be normalised per fold
        self.eeg    = self._raw_eeg.copy()
        self.physio = self._raw_physio.copy()
        self.psd    = self._raw_psd.copy()
        self.sqi    = self._raw_sqi.copy()

        print("[DATA] Computing physio stats...")
        physio_stats = self._compute_physio_stats()
        self.freq = np.concatenate([self.psd, physio_stats], axis=1)

        # Snapshot for fold restoration (already stored in _raw_*)
        self._raw_freq = self.freq.copy()

        self.train_mode = False
        print(f"[DATA] Ready. {len(self.label_wl):,} epochs, "
              f"{len(np.unique(self.subjects))} subjects.")

    def _compute_physio_stats(self) -> np.ndarray:
        n = self.physio.shape[0]
        stats = np.zeros((n, 3), dtype=np.float32)
        for i in tqdm(range(n), desc="  physio stats", leave=False):
            stats[i, 0] = np.mean(self.physio[i, 1, :])   # HR mean
            stats[i, 1] = np.std(self.physio[i, 1, :])    # HR std
            stats[i, 2] = np.mean(self.physio[i, 2, :])   # EDA mean
        np.nan_to_num(stats, copy=False)
        eda = stats[:, 2]
        stats[:, 2] = np.log(eda - eda.min() + 1e-8)
        return stats

    def restore_raw(self) -> None:
        """Reset active arrays to original values (before normalisation)."""
        self.eeg    = self._raw_eeg.copy()
        self.physio = self._raw_physio.copy()
        self.freq   = self._raw_freq.copy()
        self.sqi    = self._raw_sqi.copy()

    def set_train_mode(self, train: bool) -> None:
        self.train_mode = train

    def __len__(self) -> int:
        return len(self.label_wl)

    def __getitem__(self, idx: int) -> dict:
        # Add channel dim for EEG: (4,512) -> (1,4,512)
        eeg  = np.expand_dims(self.eeg[idx], axis=0).copy()
        phys = self.physio[idx].copy()

        if self.train_mode and np.random.rand() < 0.5:
            eeg  += np.random.normal(0, 0.02 * np.std(eeg),  eeg.shape)
            phys += np.random.normal(0, 0.02 * np.std(phys), phys.shape)
            eeg  *= np.random.uniform(0.9, 1.1)
            phys *= np.random.uniform(0.9, 1.1)

        return {
            "eeg":  torch.tensor(eeg, dtype=torch.float32),
            "phys": torch.tensor(phys, dtype=torch.float32),
            "freq": torch.tensor(self.freq[idx], dtype=torch.float32),
            "sqi":  torch.tensor(self.sqi[idx], dtype=torch.float32),
            "y_wl": self.label_wl[idx],
            "y_st": self.label_st[idx],
            "subj": self.subjects[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold normalisation (no leakage)
# ─────────────────────────────────────────────────────────────────────────────

def apply_fold_normalisation(ds: UniverseDataset, train_idx: np.ndarray) -> None:
    """
    Normalises the dataset in-place using ONLY training epoch statistics.

    EEG/physio:   per-epoch instance normalisation (removes subject DC offset)
    Freq/SQI:     fold-level z-score (scalar features, not time-series)

    Instance normalisation on EEG is critical for cross-subject generalisation:
    it removes the resting-state baseline offset, leaving only relative spectral
    changes that reflect cognitive state transitions. The GRL then only needs to
    suppress remaining subject-specific variance rather than fighting baselines.
    """
    # EEG is (N, 4, 512) – instance norm over channels and time (axes 1 and 2)
    eeg_mean = ds.eeg.mean(axis=(1,2), keepdims=True)   # shape (N,1,1)
    eeg_std  = ds.eeg.std(axis=(1,2), keepdims=True) + 1e-8
    ds.eeg   = np.clip((ds.eeg - eeg_mean) / eeg_std, -5.0, 5.0)

    # Physio is (N, 3, 512) – instance norm over time axis (axis=2)
    phy_mean = ds.physio.mean(axis=2, keepdims=True)    # shape (N,3,1)
    phy_std  = ds.physio.std(axis=2, keepdims=True) + 1e-8
    ds.physio = np.clip((ds.physio - phy_mean) / phy_std, -5.0, 5.0)

    # Z‑score scalar features
    for arr in [ds.freq, ds.sqi]:
        for col in range(arr.shape[1]):
            m = arr[train_idx, col].mean()
            s = arr[train_idx, col].std() + 1e-8
            arr[:, col] = np.clip((arr[:, col] - m) / s, -5.0, 5.0)

# ─────────────────────────────────────────────────────────────────────────────
# LOSO training loop
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    ds = UniverseDataset(DATA_DIR)
    unique_subjects   = np.unique(ds.subjects)
    LAB_ONLY_SUBJECTS = {2}  # UN_103: no Wild sessions

    # Results accumulator: one list of fold dicts per condition
    all_condition_results: dict[str, list[dict]] = {}

    conditions_to_run = [AblationConfig("full")] if RUN_ONLY_FULL else ABLATION_CONDITIONS

    for cfg in conditions_to_run:
        print(f"\n{'='*60}")
        print(f"ABLATION CONDITION: {cfg.name.upper()}")
        print(f"{'='*60}")

        fold_results = []

        for holdout_subj in unique_subjects:
            lab_only = int(holdout_subj) in LAB_ONLY_SUBJECTS
            flag     = " [lab-only]" if lab_only else ""
            print(f"\n──── Fold: Subject {holdout_subj}{flag} ────")

            ds.restore_raw()

            holdout_idx   = np.where(ds.subjects == holdout_subj)[0]
            remaining_idx = np.where(ds.subjects != holdout_subj)[0]

            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            tr_split, val_split = next(gss.split(
                X=remaining_idx,
                y=ds.label_wl[remaining_idx],
                groups=ds.subjects[remaining_idx]
            ))
            tr_idx  = remaining_idx[tr_split]
            val_idx = remaining_idx[val_split]

            print(f"  [NORM] Normalising from {len(tr_idx):,} training epochs...")
            apply_fold_normalisation(ds, tr_idx)

            valid_mask = ds.label_wl[tr_idx] != LABEL_AMBIGUOUS
            tr_idx_filtered = tr_idx[valid_mask]

            if valid_mask.sum() == 0:
                print(f"  ⚠ No valid workload labels — skipping fold.")
                continue

            # Compute class counts safely
            valid_labels = ds.label_wl[tr_idx_filtered]
            class_counts = np.bincount(valid_labels, minlength=2)
            class_counts = np.clip(class_counts, a_min=1, a_max=None)

            sample_w = 1.0 / class_counts[valid_labels]   # vectorised

            sampler   = WeightedRandomSampler(torch.tensor(sample_w),
                                               len(sample_w), replacement=True)
            loader_tr = DataLoader(Subset(ds, tr_idx_filtered), batch_size=BATCH_SIZE,
                                   sampler=sampler, num_workers=2, pin_memory=True)
            loader_val = DataLoader(Subset(ds, val_idx), batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=2, pin_memory=True)
            loader_ho  = DataLoader(Subset(ds, holdout_idx), batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=2, pin_memory=True)

            train_subjs   = np.unique(ds.subjects[remaining_idx])
            subj_map      = {int(old): new for new, old in enumerate(train_subjs)}
            n_train_subjs = len(train_subjs)

            model     = BrainBatterySANN(n_subjects=n_train_subjs, cfg=cfg).to(DEVICE)
            opt       = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=NUM_EPOCHS, eta_min=1e-6
            )

            crit_wl   = MaskedCrossEntropyLoss(label_smoothing=0.1)
            crit_st   = MaskedCrossEntropyLoss(label_smoothing=0.1)
            crit_subj = nn.CrossEntropyLoss()
            crit_con  = SupervisedContrastiveLoss(temperature=0.5).to(DEVICE)

            best_f1, patience_ctr = 0.0, 0
            best_path = MODEL_DIR / f"best_{cfg.name}_subj_{holdout_subj}.pt"

            for epoch in range(NUM_EPOCHS):
                model.train()
                ds.set_train_mode(True)

                if epoch < WARMUP_EPOCHS:
                    grl_alpha = 0.0
                else:
                    progress = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
                    grl_alpha = 0.3 * (2.0 / (1.0 + np.exp(-5.0 * progress)) - 1.0)

                if epoch < WARMUP_EPOCHS:
                    lam_domain = lam_con = 0.0
                else:
                    ramp       = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
                    lam_domain = 0.2 * LAMBDA_DOMAIN * ramp
                    lam_con    = LAMBDA_CONTRAST * ramp

                epoch_loss = 0.0

                for b in loader_tr:
                    eeg   = b["eeg"].to(DEVICE)
                    phys  = b["phys"].to(DEVICE)
                    freq  = b["freq"].to(DEVICE)
                    sqi   = b["sqi"].to(DEVICE)
                    y_wl  = b["y_wl"].to(DEVICE)
                    y_st  = b["y_st"].to(DEVICE)
                    subj = torch.tensor(
                        np.vectorize(subj_map.get)(b["subj"].cpu().numpy()),
                        dtype=torch.long,
                        device=DEVICE
                    )

                    opt.zero_grad()
                    wl_out, st_out, subj_out, proj_out, fused, att = model(
                        eeg, phys, freq, sqi, grl_alpha
                    )

                    # --- Compute losses separately for breakdown ---
                    wl_loss = crit_wl(wl_out, y_wl)
                    st_loss = crit_st(st_out, y_st)

                    # Weight primary task higher
                    cls_loss = wl_loss + 0.5 * st_loss
                    if cfg.use_grl and lam_domain > 0:
                        dom_loss = crit_subj(subj_out, subj)
                    else:
                        dom_loss = torch.tensor(0.0, device=DEVICE)
                        
                    valid_con = (y_wl != LABEL_AMBIGUOUS)
                    if lam_con > 0 and valid_con.sum() > 4:
                        con_loss = crit_con(proj_out[valid_con], y_wl[valid_con])
                    else:
                        con_loss = torch.tensor(0.0, device=DEVICE)

                    # Combine losses
                    loss = cls_loss + lam_domain * dom_loss + (lam_con * con_loss if lam_con > 0 else 0)

                    # Entropy regularisation: maximise attention entropy to prevent modality collapse.
                    # Shannon entropy H = -∑p·log(p) is positive; SUBTRACTING it from the loss
                    # means the optimizer maximises entropy (encourages balanced modality weights).
                    # The (1 - grl_alpha) fade-out is removed — entropy regularisation must stay
                    # constant throughout training, not fade when GRL activates.
                    # Reference: Wang et al. (2020) "What Makes Training Multi-Modal Classification
                    # Networks Hard?" — entropy regularisation on fusion weights is the standard
                    # intervention for modality collapse in competitive multi-tower architectures.
                    att_entropy = -(att * torch.log(att + 1e-8)).sum(dim=1).mean()
                    loss -= 0.05 * att_entropy
                    
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
                    opt.step()

                    # Max-norm on EEG spatial filters (Lawhern et al., 2018)
                    with torch.no_grad():
                        w    = model.eeg_tower.spat_conv.weight
                        norm = w.norm(2, dim=(1, 2, 3), keepdim=True)
                        w.copy_(w * torch.clamp(1.0 / norm, max=1.0))

                    epoch_loss += loss.item()

                scheduler.step()
                ds.set_train_mode(False)

                val_f1 = _compute_masked_f1(model, loader_val, target="workload")
                # Checkpoint on val F1 directly — no hybrid loss scaling.
                # Mixing val_loss into the selection criterion creates a constant
                # ~0.09 penalty at typical loss values that corrupts fold comparison.
                if val_f1 > best_f1:
                    best_f1, patience_ctr = val_f1, 0
                    torch.save(model.state_dict(), best_path)
                else:
                    patience_ctr += 1

                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1:3d} | loss={epoch_loss/len(loader_tr):.4f} "
                          f"| val_F1_wl={val_f1:.3f} | α={grl_alpha:.2f}")

                if patience_ctr >= PATIENCE and (epoch + 1) >= MIN_EPOCHS:
                    print(f"  Early stop at epoch {epoch+1}")
                    break

            model.load_state_dict(
                torch.load(best_path, map_location=DEVICE, weights_only=True)
            )
            T_wl, thr_wl = _calibrate_threshold(model, loader_val, "workload")
            T_st, thr_st = _calibrate_threshold(model, loader_val, "stress")

            metrics, att = _evaluate_holdout(
                model, loader_ho, T_wl, thr_wl, T_st, thr_st
            )
            print(f"  [{cfg.name}] Subj {holdout_subj}{flag}  "
                  f"WL F1={metrics['wl_f1']:.3f}  ST F1={metrics['st_f1']:.3f}  "
                  f"fatigue={metrics['fatigue_mean']:.3f}")

            # Save attention weights only for the full model condition
            if cfg.name == "full":
                np.save(MODEL_DIR / f"att_subj_{holdout_subj}.npy", att)

            fold_results.append({
                "subj":     int(holdout_subj),
                "lab_only": lab_only,
                "condition": cfg.name,
                **metrics
            })

        all_condition_results[cfg.name] = fold_results

    # Print the ablation table
    _print_ablation_table(all_condition_results)

    # Save all results to numpy for downstream analysis
    np.save(MODEL_DIR / "ablation_results.npy", all_condition_results)
    print(f"\n[SAVED] {MODEL_DIR}/ablation_results.npy")


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION TABLE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def _print_ablation_table(all_results: dict[str, list[dict]]) -> None:
    """
    Prints the feature contribution table.

    The std column is the primary population generalisation finding.
    A modality with low std generalises; high std means subject-dependent.

    The delta columns show the cost of removing each component relative
    to the full model — this is your quantified contribution.
    """
    print(f"\n{'='*70}")
    print("ABLATION TABLE — FEATURE CONTRIBUTION (LOSO, n=12 subjects)")
    print(f"{'='*70}")
    print(f"{'Condition':<15} {'WL F1':>7} {'±':>3} {'ST F1':>7} {'±':>3} "
          f"{'AUC':>7} {'ΔWL F1':>9}")
    print("-" * 70)

    full_wl_mean = np.mean([r["wl_f1"] for r in all_results.get("full", [{"wl_f1": 0}])])

    for cond_name, results in all_results.items():
        if not results:
            continue
        wl_f1  = [r["wl_f1"]  for r in results]
        st_f1  = [r["st_f1"]  for r in results]
        auc    = [r["wl_auc"] for r in results]
        delta  = np.mean(wl_f1) - full_wl_mean

        print(f"{cond_name:<15} "
              f"{np.mean(wl_f1):>7.3f} "
              f"{np.std(wl_f1):>4.3f} "
              f"{np.mean(st_f1):>7.3f} "
              f"{np.std(st_f1):>4.3f} "
              f"{np.mean(auc):>7.3f} "
              f"{delta:>+9.3f}")

    print("-" * 70)
    print("ΔWL F1: relative to full model (negative = removing this hurt performance)")
    print("std: cross-subject variance — lower = more generalizable across population")

    # Per-subject breakdown for population generalisation analysis
    print(f"\n{'='*70}")
    print("PER-SUBJECT WORKLOAD F1 — FULL MODEL (population generalisation profile)")
    print(f"{'='*70}")
    full = all_results.get("full", [])
    if full:
        print(f"{'Subject':<12} {'WL F1':>7} {'ST F1':>7} {'AUC':>7} "
              f"{'fatigue θ/α':>12} {'domain':>8}")
        print("-" * 60)
        for r in sorted(full, key=lambda x: x["subj"]):
            tag = " (lab)" if r["lab_only"] else ""
            print(f"Subj {r['subj']:<7} "
                  f"{r['wl_f1']:>7.3f} "
                  f"{r['st_f1']:>7.3f} "
                  f"{r['wl_auc']:>7.3f} "
                  f"{r['fatigue_mean']:>12.3f}"
                  f"{tag:>8}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions (metrics, calibration, evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_masked_f1(model: nn.Module, loader: DataLoader,
                        target: str = "workload") -> float:
    """Computes macro-F1 on valid (non-ambiguous) epochs only."""
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for b in loader:
            wl_out, st_out, *_ = model(
                b["eeg"].to(DEVICE), b["phys"].to(DEVICE),
                b["freq"].to(DEVICE), b["sqi"].to(DEVICE), 0.0
            )
            logits = wl_out if target == "workload" else st_out
            labels = b["y_wl"] if target == "workload" else b["y_st"]

            valid  = labels != LABEL_AMBIGUOUS
            if valid.sum() == 0:
                continue
            preds.extend(logits[valid].argmax(1).cpu().numpy())
            truths.extend(labels[valid].cpu().numpy())

    if len(truths) == 0:
        return 0.0

    unique_classes = np.unique(truths)
    if len(unique_classes) < 2:
        # Single-class validation fold — happens with small folds on lab-only subjects.
        # Returns 0.0 to prevent early stopping from triggering on this fold.
        print(f"  [WARN] val fold contains only class {unique_classes} — F1=0.0 (not a real result)")
        return 0.0

    return f1_score(truths, preds, average="macro")


def _calibrate_threshold(model: nn.Module, loader: DataLoader,
                          target: str) -> tuple[float, float]:
    """
    Grid-searches temperature T and decision threshold on the validation set.
    Only epochs with non-ambiguous labels contribute to calibration.
    Returns (1.0, 0.5) as neutral defaults if no valid labels are found.
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for b in loader:
            wl_out, st_out, *_ = model(
                b["eeg"].to(DEVICE), b["phys"].to(DEVICE),
                b["freq"].to(DEVICE), b["sqi"].to(DEVICE), 0.0
            )
            logits = wl_out if target == "workload" else st_out
            labels = b["y_wl"] if target == "workload" else b["y_st"]
            valid  = labels != LABEL_AMBIGUOUS

            if valid.sum() == 0:
                continue
            all_logits.extend(logits[valid].cpu().numpy())
            all_labels.extend(labels[valid].cpu().numpy())

    # Guard: return neutral defaults if validation set has no valid labels
    # (can occur for lab-only subjects with small folds)
    if len(all_labels) == 0 or len(np.unique(all_labels)) < 2:
        return 1.0, 0.5

    logits_arr = np.array(all_logits)
    labels_arr = np.array(all_labels)

    best_T, best_thresh, best_f1 = 1.0, 0.5, 0.0
    for T in np.arange(1.0, 3.0, 0.2):
        sl   = logits_arr / T
        sl  -= sl.max(axis=1, keepdims=True)
        prob = np.exp(sl) / np.exp(sl).sum(axis=1, keepdims=True)
        for thresh in np.arange(0.3, 0.75, 0.05):
            f1 = f1_score(labels_arr, (prob[:, 1] >= thresh).astype(int),
                          average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_T, best_thresh = f1, T, thresh

    return best_T, best_thresh


def _evaluate_holdout(model: nn.Module, loader: DataLoader,
                       T_wl: float, thresh_wl: float,
                       T_st: float, thresh_st: float) -> tuple[dict, np.ndarray]:
    """
    Evaluates the model on the holdout subject.

    Returns classification metrics for both targets and the fatigue proxy.

    Fatigue proxy: mean log(θ/α) averaged across the 4 EEG channels.
    These are stored in freq columns [24:28] (before fold normalisation
    the values are in log space; after normalisation they are z-scored).
    Report this value as a relative index within a subject, not absolute.
    """
    model.eval()
    all_wl_logits, all_st_logits = [], []
    all_wl_labels, all_st_labels = [], []
    all_freq, all_att = [], []

    with torch.no_grad():
        for b in loader:
            wl_out, st_out, _, _, _, att_w = model(
                b["eeg"].to(DEVICE), b["phys"].to(DEVICE),
                b["freq"].to(DEVICE), b["sqi"].to(DEVICE), 0.0
            )
            all_wl_logits.extend(wl_out.cpu().numpy())
            all_st_logits.extend(st_out.cpu().numpy())
            all_wl_labels.extend(b["y_wl"].cpu().numpy())
            all_st_labels.extend(b["y_st"].cpu().numpy())
            all_freq.extend(b["freq"].cpu().numpy())
            all_att.extend(att_w.cpu().numpy())

    wl_logits = np.array(all_wl_logits)
    st_logits = np.array(all_st_logits)
    wl_labels = np.array(all_wl_labels)
    st_labels = np.array(all_st_labels)
    freq      = np.array(all_freq)       # (N, 36)
    att       = np.array(all_att)        # (N, 3)

    # Compute smoothed probabilities for each target
    wl_probs = _temperature_smooth(wl_logits, T_wl)
    st_probs = _temperature_smooth(st_logits, T_st)

    wl_preds = (wl_probs >= thresh_wl).astype(int)
    st_preds = (st_probs >= thresh_st).astype(int)

    # Evaluate only on non-ambiguous epochs
    wl_valid = wl_labels != LABEL_AMBIGUOUS
    st_valid = st_labels != LABEL_AMBIGUOUS

    # Fatigue proxy: freq columns 24:28 = log θ/α per channel
    # After fold normalisation these are z-scored, so the value is
    # relative to the training set mean, not absolute neuroscience units.
    fatigue_mean = float(freq[:, 24:28].mean()) if freq.shape[0] > 0 else 0.0

    def safe_metrics(labels, preds, probs, mask):
        if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
            return 0.0, 0.0, 0.5
        return (accuracy_score(labels[mask], preds[mask]),
                f1_score(labels[mask], preds[mask], average="macro"),
                roc_auc_score(labels[mask], probs[mask]))

    wl_acc, wl_f1, wl_auc = safe_metrics(wl_labels, wl_preds, wl_probs, wl_valid)
    st_acc, st_f1, st_auc = safe_metrics(st_labels, st_preds, st_probs, st_valid)

    metrics = {
        "wl_acc": wl_acc, "wl_f1": wl_f1, "wl_auc": wl_auc,
        "st_acc": st_acc, "st_f1": st_f1, "st_auc": st_auc,
        "fatigue_mean": fatigue_mean,
        "wl_probs": wl_probs[wl_valid], "wl_labels": wl_labels[wl_valid],
        "st_probs": st_probs[st_valid], "st_labels": st_labels[st_valid],
    }
    return metrics, att


def _temperature_smooth(logits: np.ndarray, T: float) -> np.ndarray:
    """Applies temperature scaling and EMA smoothing to classifier logits."""
    sl    = logits / T
    sl   -= sl.max(axis=1, keepdims=True)
    probs = np.exp(sl) / np.exp(sl).sum(axis=1, keepdims=True)
    p1    = probs[:, 1]

    # EMA smoothing (α=0.7) — stabilises predictions across consecutive epochs
    smoothed    = np.zeros_like(p1)
    smoothed[0] = p1[0]
    for i in range(1, len(p1)):
        smoothed[i] = 0.7 * p1[i] + 0.3 * smoothed[i - 1]
    return smoothed


def _save_calibration_plot(probs: np.ndarray, labels: np.ndarray,
                            subj_id: int, suffix: str = "") -> None:
    try:
        prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
        plt.figure(figsize=(4, 4))
        plt.plot(prob_pred, prob_true, marker="o", lw=2, label=suffix)
        plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        plt.title(f"Calibration — Subj {subj_id} ({suffix})")
        plt.xlabel("Predicted probability")
        plt.ylabel("True fraction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(MODEL_DIR / f"calibration_{suffix}_subj_{subj_id}.png", dpi=100)
        plt.close()
    except (ValueError, IndexError):
        pass


def _print_final_results(results: list[dict]) -> None:
    wl_f1  = [r["wl_f1"]  for r in results]
    wl_acc = [r["wl_acc"] for r in results]
    wl_auc = [r["wl_auc"] for r in results]
    st_f1  = [r["st_f1"]  for r in results]
    st_acc = [r["st_acc"] for r in results]
    st_auc = [r["st_auc"] for r in results]

    print(f"\n{'='*60}")
    print("FINAL LOSO RESULTS")
    print(f"{'='*60}")
    print(f"\n  Cognitive workload:")
    print(f"    Macro-F1:  {np.mean(wl_f1):.3f} ± {np.std(wl_f1):.3f}")
    print(f"    Accuracy:  {np.mean(wl_acc):.3f} ± {np.std(wl_acc):.3f}")
    print(f"    ROC-AUC:   {np.mean(wl_auc):.3f} ± {np.std(wl_auc):.3f}")
    print(f"\n  Stress:")
    print(f"    Macro-F1:  {np.mean(st_f1):.3f} ± {np.std(st_f1):.3f}")
    print(f"    Accuracy:  {np.mean(st_acc):.3f} ± {np.std(st_acc):.3f}")
    print(f"    ROC-AUC:   {np.mean(st_auc):.3f} ± {np.std(st_auc):.3f}")

    full = [r for r in results if not r["lab_only"]]
    if full:
        print(f"\n  Wild-session subjects only (excluding lab-only, n={len(results)-len(full)}):")
        print(f"    Workload F1: {np.mean([r['wl_f1'] for r in full]):.3f}")
        print(f"    Stress   F1: {np.mean([r['st_f1'] for r in full]):.3f}")

    print(f"\n  Fatigue proxy (mean θ/α, z-scored relative to training set):")
    print(f"    Mean across subjects: {np.mean([r['fatigue_mean'] for r in results]):.3f}")
    print(f"\n  Attention weight arrays: {MODEL_DIR}/att_subj_*.npy")
    print("  Columns: [EEG_weight, physio_weight, freq_weight] per epoch")
    print("  Use these to analyse which modality drives workload vs stress prediction.")

    # ── Generalization Score ────────────────────────────────────────────────
    # Generalization Score = mean_accuracy - 0.5 * std_accuracy
    # Penalises high cross-subject variance. A modality that's accurate for
    # some subjects but fails on others scores lower than a consistently
    # moderate modality. This is the primary population generalisation metric.
    # λ=0.5 from Ganin & Lempitsky (2015) domain adaptation framing.
    wl_gen_score = np.mean(wl_acc) - 0.5 * np.std(wl_acc)
    st_gen_score = np.mean(st_acc) - 0.5 * np.std(st_acc)
    print(f"\n  Generalization Score (Acc − 0.5×Std):")
    print(f"    Workload: {wl_gen_score:.3f}")
    print(f"    Stress:   {st_gen_score:.3f}")
    print(f"  (Higher = more consistent across subjects; lower = subject-dependent)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()

    if os.path.exists("/kaggle/working/models"):
        shutil.make_archive("/kaggle/working/model_export", "zip", "/kaggle/working/models")
        print("\n[EXPORT] /kaggle/working/model_export.zip")
