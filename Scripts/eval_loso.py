"""
eval_loso.py — Leave-One-Subject-Out cross-validation for Brain Battery.

Runs the pre-trained MultimodalCWSANN checkpoint against all 24 subjects and
compares against an XGBoost baseline trained on pre-computed spectral features.
Saves per-subject results to results/loso_results.json and confusion matrix
PNGs to results/plots/.

Usage:
    python scripts/eval_loso.py

No Streamlit required. Runs in ~10-20 minutes on CPU.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# ── Paths (relative to repo root) ────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "Data"
MODEL_PATH = ROOT / "model" / "best_full_subj_17.pt"
RESULTS    = ROOT / "results"
PLOTS_DIR  = RESULTS / "plots"
RESULTS.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ── Device (DirectML → CUDA → CPU) ───────────────────────────────────────────
try:
    import torch_directml
    DEVICE = torch_directml.device()
    print("Device: AMD DirectML (RX 6800)")
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

# ── Architecture constants (must match checkpoint) ────────────────────────────
SQI_DIM    = 6
TOWER_DIM  = 32
FUSION_DIM = 96
BATCH_SIZE = 256


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE  (mirrors dashboard.py — keep in sync)
# ══════════════════════════════════════════════════════════════════════════════

class SQIGate(nn.Module):
    def __init__(self, sqi_dim=SQI_DIM):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(sqi_dim, 16), nn.ReLU(),
                                   nn.Linear(16, 3), nn.Sigmoid())
    def forward(self, sqi): return self.proj(sqi)


class AttentionFusion(nn.Module):
    def __init__(self, in_dim=FUSION_DIM, sqi_dim=SQI_DIM):
        super().__init__()
        self.content_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.ReLU(), nn.Linear(in_dim // 2, 3))
        self.sqi_gate = SQIGate(sqi_dim)
        self.norm     = nn.LayerNorm(in_dim)

    def forward(self, f_eeg, f_phys, f_freq, sqi):
        concat  = torch.cat([f_eeg, f_phys, f_freq], dim=1)
        logits  = self.content_head(concat)
        quality = self.sqi_gate(sqi)
        att     = F.softmax(logits * (quality + 0.1) / 3.0, dim=1)
        fused   = torch.cat([att[:,0:1]*f_eeg, att[:,1:2]*f_phys, att[:,2:3]*f_freq], dim=1)
        return self.norm(fused), att


class EEGNet(nn.Module):
    def __init__(self, n_ch=4, F1=16, D=2, F2=32, dropout=0.5):
        super().__init__()
        self.temp_conv  = nn.Conv2d(1, F1, (1, 128), padding=(0, 64), bias=False)
        self.bn1        = nn.BatchNorm2d(F1)
        self.spat_conv  = nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False)
        self.bn2        = nn.BatchNorm2d(F1*D); self.elu = nn.ELU()
        self.pool1      = nn.AvgPool2d((1, 4)); self.drop1 = nn.Dropout(dropout)
        self.sep_conv   = nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3        = nn.BatchNorm2d(F2)
        self.pool2      = nn.AvgPool2d((1, 8)); self.drop2 = nn.Dropout(dropout)
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(F2, F2//4, 1), nn.ReLU(),
                                 nn.Conv2d(F2//4, F2, 1), nn.Sigmoid())
    def forward(self, x):
        x = self.bn1(self.temp_conv(x))
        x = self.drop1(self.pool1(self.elu(self.bn2(self.spat_conv(x)))))
        x = self.drop2(self.pool2(self.elu(self.bn3(self.sep_conv(x)))))
        return self.adapt_pool(x * self.se(x)).view(x.size(0), -1)


class PhysioCNN(nn.Module):
    def __init__(self, out_dim=TOWER_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 8, 15, padding=7), nn.BatchNorm1d(8), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(8, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(16, out_dim); self.drop = nn.Dropout(0.3)
    def forward(self, x): return self.fc(self.drop(self.net(x).squeeze(-1)))


class FreqEncoder(nn.Module):
    def __init__(self, in_dim=36, out_dim=TOWER_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.BatchNorm1d(64),
                                  nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))
    def forward(self, x): return self.net(x)


class MultimodalCWSANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.eeg_tower     = EEGNet(n_ch=4, F1=16, D=2, F2=TOWER_DIM, dropout=0.5)
        self.physio_tower  = PhysioCNN(out_dim=TOWER_DIM)
        self.freq_tower    = FreqEncoder(in_dim=36, out_dim=TOWER_DIM)
        self.fusion        = AttentionFusion(in_dim=FUSION_DIM, sqi_dim=SQI_DIM)
        self.workload_head = nn.Sequential(
            nn.Linear(FUSION_DIM, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, eeg, physio, freq, sqi=None):
        if sqi is None:
            sqi = torch.full((eeg.size(0), SQI_DIM), 0.5, device=eeg.device)
        fused, att = self.fusion(self.eeg_tower(eeg), self.physio_tower(physio),
                                  self.freq_tower(freq), sqi)
        return self.workload_head(fused), att


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load all arrays with memory-mapping. Returns dict of numpy arrays."""
    print("Loading data arrays...")
    required = ["eeg.npy", "physio.npy", "label_workload.npy", "subjects.npy"]
    for f in required:
        if not (DATA_DIR / f).exists():
            sys.exit(f"Missing: {DATA_DIR / f}  —  run the app first to download data.")

    data = {
        "eeg":      np.load(DATA_DIR / "eeg.npy",           mmap_mode="r"),
        "physio":   np.load(DATA_DIR / "physio.npy",         mmap_mode="r"),
        "labels":   np.load(DATA_DIR / "label_workload.npy", mmap_mode="r"),
        "subjects": np.load(DATA_DIR / "subjects.npy",       mmap_mode="r"),
    }
    # Pre-computed spectral features (36-dim) — skip live Welch recomputation
    psd_path = DATA_DIR / "psd_features.npy"
    data["psd"] = np.load(psd_path, mmap_mode="r") if psd_path.exists() else None

    n = len(data["labels"])
    unique_subjects = sorted(np.unique(data["subjects"]).tolist())
    print(f"  {n:,} epochs · {len(unique_subjects)} subjects · "
          f"PSD features: {'yes' if data['psd'] is not None else 'will recompute'}")
    return data, unique_subjects


def normalize(arr: np.ndarray) -> np.ndarray:
    """Instance normalization per channel, clipped to ±5 (Schirrmeister 2017)."""
    mu = arr.mean(axis=-1, keepdims=True)
    sd = arr.std(axis=-1,  keepdims=True) + 1e-6
    return np.clip((arr - mu) / sd, -5.0, 5.0)


def load_model() -> MultimodalCWSANN:
    if not MODEL_PATH.exists():
        sys.exit(f"Checkpoint not found: {MODEL_PATH}")
    model = MultimodalCWSANN()
    ckpt  = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    expected_missing = {"stress_head", "subject_head", "projector"}
    result = model.load_state_dict(ckpt, strict=False)
    critical = [k for k in result.missing_keys
                if not any(x in k for x in expected_missing)]
    if critical:
        sys.exit(f"Architecture mismatch — missing keys: {critical[:5]}")
    return model.to(DEVICE).eval()


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_sann(model, eeg_np, physio_np, psd_np):
    """Batch inference. Returns (probs, preds) as numpy arrays."""
    N = len(eeg_np)
    all_probs, all_preds = [], []
    for start in range(0, N, BATCH_SIZE):
        sl = slice(start, start + BATCH_SIZE)
        eeg_b    = torch.tensor(eeg_np[sl],    dtype=torch.float32).unsqueeze(1).to(DEVICE)
        physio_b = torch.tensor(physio_np[sl], dtype=torch.float32).to(DEVICE)
        freq_b   = torch.tensor(psd_np[sl],   dtype=torch.float32).to(DEVICE)
        logits, _ = model(eeg_b, physio_b, freq_b)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_probs.append(probs); all_preds.append(preds)
    return np.concatenate(all_probs), np.concatenate(all_preds)


# ══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX PLOT
# ══════════════════════════════════════════════════════════════════════════════

def save_confusion(cm, subject_id, acc, model_name):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor("#0D0F14")
    ax.set_facecolor("#0D0F14")
    im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
    ax.set_title(f"S{subject_id:02d} — {model_name}  (acc {acc:.1%})",
                 color="white", fontsize=10, pad=8)
    ax.set_xlabel("Predicted", color="#9CA3AF"); ax.set_ylabel("Actual", color="#9CA3AF")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low CW", "High CW"], color="white", fontsize=8)
    ax.set_yticklabels(["Low CW", "High CW"], color="white", fontsize=8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] < cm.max() * 0.6 else "black", fontsize=12)
    fig.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color="white")
    plt.tight_layout()
    tag = model_name.lower().replace(" ", "_")
    path = PLOTS_DIR / f"subject_{subject_id:02d}_{tag}_confusion.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOSO LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_loso():
    data, subjects = load_data()
    model          = load_model()
    print(f"\nRunning LOSO on {len(subjects)} subjects...\n")

    per_subject = []
    sann_accs, xgb_accs = [], []

    for s in subjects:
        test_mask  = data["subjects"] == s
        train_mask = ~test_mask

        # ── Slice test arrays ─────────────────────────────────────────────
        t0 = time.perf_counter()
        eeg_test    = normalize(np.array(data["eeg"][test_mask],    dtype=np.float32))
        physio_test = normalize(np.array(data["physio"][test_mask], dtype=np.float32))
        labels_test = np.array(data["labels"][test_mask], dtype=np.int32)
        psd_test    = np.array(data["psd"][test_mask],    dtype=np.float32) \
                      if data["psd"] is not None else None
        t_load = time.perf_counter() - t0

        n_test = len(labels_test)
        if n_test == 0:
            print(f"  S{s:02d}: no test epochs — skipping"); continue

        # ── SANN inference ────────────────────────────────────────────────
        if psd_test is None:
            print(f"  S{s:02d}: psd_features.npy missing — skipping SANN freq tower")
            sann_result = None
        else:
            t1 = time.perf_counter()
            sann_probs, sann_preds = run_sann(model, eeg_test, physio_test, psd_test)
            t_infer = time.perf_counter() - t1

            sann_acc = accuracy_score(labels_test, sann_preds)
            sann_f1  = f1_score(labels_test, sann_preds, zero_division=0)
            try:
                sann_auc = roc_auc_score(labels_test, sann_probs)
            except ValueError:
                sann_auc = float("nan")

            cm_sann = confusion_matrix(labels_test, sann_preds, labels=[0, 1])
            cm_path = save_confusion(cm_sann, s, sann_acc, "SANN")

            # Bottleneck flag
            bottleneck = ""
            if t_infer > 0 and (t_load / t_infer) > 1.2:
                bottleneck = "CPU_BOTTLENECK: data load > inference — consider pre-pinning tensors"

            sann_result = {
                "acc": round(sann_acc, 4), "f1": round(sann_f1, 4),
                "auc": round(sann_auc, 4),
                "t_load_ms": round(t_load * 1000, 1),
                "t_infer_ms": round(t_infer * 1000, 1),
                "bottleneck": bottleneck,
                "confusion_matrix": cm_sann.tolist(),
                "confusion_plot": cm_path,
            }
            sann_accs.append(sann_acc)

        # ── XGBoost baseline ──────────────────────────────────────────────
        if data["psd"] is not None:
            psd_train  = np.array(data["psd"][train_mask],     dtype=np.float32)
            labels_train = np.array(data["labels"][train_mask], dtype=np.int32)

            t2 = time.perf_counter()
            xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                 eval_metric="logloss", verbosity=0,
                                 use_label_encoder=False)
            xgb.fit(psd_train, labels_train)
            xgb_preds = xgb.predict(psd_test)
            xgb_probs = xgb.predict_proba(psd_test)[:, 1]
            t_xgb = time.perf_counter() - t2

            xgb_acc = accuracy_score(labels_test, xgb_preds)
            xgb_f1  = f1_score(labels_test, xgb_preds, zero_division=0)
            try:
                xgb_auc = roc_auc_score(labels_test, xgb_probs)
            except ValueError:
                xgb_auc = float("nan")

            cm_xgb  = confusion_matrix(labels_test, xgb_preds, labels=[0, 1])
            cm_path_xgb = save_confusion(cm_xgb, s, xgb_acc, "XGBoost")

            xgb_result = {
                "acc": round(xgb_acc, 4), "f1": round(xgb_f1, 4),
                "auc": round(xgb_auc, 4),
                "t_xgb_ms": round(t_xgb * 1000, 1),
                "confusion_matrix": cm_xgb.tolist(),
                "confusion_plot": cm_path_xgb,
            }
            xgb_accs.append(xgb_acc)
        else:
            xgb_result = None

        per_subject.append({
            "subject": int(s),
            "n_test_epochs": int(n_test),
            "sann": sann_result,
            "xgboost": xgb_result,
        })

        sann_str = f"SANN {sann_result['acc']:.1%}  F1 {sann_result['f1']:.3f}" \
                   if sann_result else "SANN skipped"
        xgb_str  = f"XGB  {xgb_result['acc']:.1%}  F1 {xgb_result['f1']:.3f}" \
                   if xgb_result else "XGB  skipped"
        flag     = f"  ⚠ {sann_result['bottleneck']}" \
                   if sann_result and sann_result.get("bottleneck") else ""
        print(f"  S{s:02d} ({n_test:4d} epochs)  {sann_str}  |  {xgb_str}{flag}")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "sann_mean_acc":  round(float(np.mean(sann_accs)),  4) if sann_accs  else None,
        "sann_std_acc":   round(float(np.std(sann_accs)),   4) if sann_accs  else None,
        "xgb_mean_acc":   round(float(np.mean(xgb_accs)),   4) if xgb_accs   else None,
        "xgb_std_acc":    round(float(np.std(xgb_accs)),    4) if xgb_accs   else None,
        "best_subject":   int(subjects[int(np.argmax(sann_accs))]) if sann_accs else None,
        "worst_subject":  int(subjects[int(np.argmin(sann_accs))]) if sann_accs else None,
        "device":         str(DEVICE),
    }
    output = {"summary": summary, "per_subject": per_subject}

    out_path = RESULTS / "loso_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'─'*60}")
    print(f"SANN  mean acc: {summary['sann_mean_acc']:.1%}  ± {summary['sann_std_acc']:.3f}")
    print(f"XGB   mean acc: {summary['xgb_mean_acc']:.1%}  ± {summary['xgb_std_acc']:.3f}")
    print(f"Best subject:  S{summary['best_subject']:02d}")
    print(f"Worst subject: S{summary['worst_subject']:02d}")
    print(f"\nResults saved to {out_path}")
    print(f"Confusion matrices saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    run_loso()
