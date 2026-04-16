"""
Brain Battery — Cognitive Bandwidth Monitor  v4.2
==================================================
WHAT THIS MEASURES
------------------
1. COGNITIVE WORKLOAD (CW)  ← hero metric
   P(high CW) = softmax(workload_head(EEG ⊕ Physio ⊕ Freq))₁
   Workload  = 100 × P(high CW)
   0% = fully rested. 100% = brain maxed out.
   NASA-TLX binary labels, UNIVERSE dataset (n=12 subjects, LOSO).
   NOT emotional stress — different construct, different labels.

2. MENTAL FATIGUE  (EEG spectral only — no model involved)
   Fatigue = log θ_power − log α_power   (Klimesch, 1999)
   θ = 4–8 Hz, α = 8–13 Hz.

3. FOCUS STREAK
   Continuous low-CW windows. Grace period: 3 consecutive HIGH windows
   required before streak resets.

4. SIGNAL QUALITY
   Live: per-channel std heuristic (Viola et al. 2009).
   Demo: spectral attention weight proxy.

5. EMOTIONAL STRESS — NOT MEASURED.
   Different construct, different labels, different biomarkers.

CORE RESEARCH CLAIM
-------------------
Spectral features generalize across subjects better than raw EEG or
physiological signals — but no modality achieves reliable zero-calibration
at the individual level.

CHANGES v4.2
------------
Task 1 — State machine fixes:
  • Subject change in sidebar now explicitly sets running=False and
    calls _reset_history() before st.rerun(). No stale loop state.
  • Restart button now calls st.rerun() immediately after resetting all
    deques — does not rely on fall-through logic.
  • acquire_data guard simplified: loop always calls st.rerun() when
    running is True (even on the first frame), so the loop never dies.

Task 2 — Bandwidth → Cognitive Workload reversal:
  • Metric: Workload = 100 × P(high CW)   (was 100×(1−P))
  • personalized_workload() replaces personalized_bandwidth() — reversal
    of the normalisation direction.
  • wl_hist replaces bw_hist throughout session state, charts, logger.
  • _wl_color() 0–30 green / 30–70 amber / 70–100 red (high = bad).
  • Pill text: "LOW EFFORT" / "ACTIVE" / "HEAVY LOAD".
  • Energy bar fills left-to-right as workload rises; labels read
    "Resting" (left) and "Overloaded" (right).

CRITICAL BUG FIX (v3.0, retained)
-----------------------------------
ZERO st.sidebar.* calls anywhere inside live_display().
Debug values written to st.session_state._dbg_*, read outside fragment.

References
----------
Welford (1962) Technometrics 4(3):419-420
Klimesch (1999) Brain Research Reviews 29:169-195
Lawhern et al. (2018) J. Neural Eng. 15(5)
Holm et al. (2009) Psychophysiology 46(5):583-598
Schirrmeister et al. (2017) Hum Brain Mapp 38(9):4456-4473
Viola et al. (2009) J Neurosci Methods 182(1):15-26
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import collections
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import welch

try:
    import pylsl
    PYLSL_AVAILABLE = True
except ImportError:
    PYLSL_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  ── edit these two paths before running locally
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR   = r"/kaggle/input/datasets/brandon19834/universe-merged-withzero-noasr"
MODEL_PATH = r"/kaggle/input/datasets/brandon19834/best-model-subj-11/best_full_subj_17.pt"

PROFILES_DIR = Path("user_profiles"); PROFILES_DIR.mkdir(exist_ok=True)
HISTORY_FILE = Path("history.jsonl")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG + GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Brain Battery",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html { overflow-anchor: none; }
.stApp { background: #0A0C10; font-family: 'DM Sans', sans-serif; }
.block-container { padding: 1.2rem 2rem 3rem 2rem !important; max-width: 1100px; }

div[data-testid="stSidebar"] {
    background: #0D0F14 !important;
    border-right: 1px solid #1A1D24;
}

h1, h2, h3 { font-family: 'DM Mono', monospace !important; }
div[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    color: #00B3CC !important;
    font-size: 1.5rem !important;
}

/* ── Cards ── */
.bb-card {
    background: #1E2329;
    border: 1px solid #2A2F3A;
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 10px;
}
.bb-card-hero {
    background: linear-gradient(135deg, #111827 0%, #1A2033 100%);
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 10px;
}
.bb-card-accent {
    background: #0D1A1C;
    border: 1px solid #00B3CC;
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 10px;
}
.bb-divider { height: 1px; background: #1A1D24; margin: 8px 0; }

/* ── Typography ── */
.bb-label {
    color: #4B5563;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: 'DM Mono', monospace;
    margin-bottom: 4px;
    display: block;
}
.bb-value {
    color: #E5E7EB;
    font-size: 36px;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    line-height: 1;
}
.bb-sub   { color: #8B949E; font-size: 12px; font-family: 'DM Sans', sans-serif; }
.bb-desc  { color: #8B949E; font-size: 13px; font-family: 'DM Sans', sans-serif; line-height: 1.6; }
.bb-title { font-family: 'DM Mono', monospace; color: #00B3CC;
            font-size: 22px; font-weight: 500; letter-spacing: -0.02em; margin: 0; }

.feed-section-title {
    font-family: 'DM Mono', monospace;
    color: #E5E7EB;
    font-size: 14px;
    font-weight: 500;
    margin: 0 0 4px 0;
}
.feed-section-cite {
    color: #4B5563;
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    margin-bottom: 8px;
    display: block;
}

/* ── Pills ── */
.pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 100px;
    font-size: 11px; font-weight: 600;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.04em; white-space: nowrap;
}
.pill-green  { background: rgba(0,200,100,0.08);  color: #00CC77; border: 1px solid rgba(0,200,100,0.2); }
.pill-orange { background: rgba(240,140,0,0.08);  color: #E09000; border: 1px solid rgba(240,140,0,0.2); }
.pill-red    { background: rgba(220,60,60,0.08);  color: #E05050; border: 1px solid rgba(220,60,60,0.2); }
.pill-blue   { background: rgba(0,179,204,0.08);  color: #00B3CC; border: 1px solid rgba(0,179,204,0.2); }
.pill-purple { background: rgba(160,110,220,0.08);color: #A06EDC; border: 1px solid rgba(160,110,220,0.2); }
.pill-dim    { background: rgba(107,114,128,0.08);color: #6B7280; border: 1px solid rgba(107,114,128,0.2); }

/* ── Energy bar (workload — fills as load rises) ── */
.energy-bar-wrap {
    background: #262B34;
    border-radius: 100px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    margin: 10px 0 6px 0;
}
.energy-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Focus streak timer ── */
.streak-display {
    font-family: 'DM Mono', monospace;
    font-size: 44px;
    font-weight: 400;
    line-height: 1;
    letter-spacing: -0.02em;
}
.streak-label { color: #4B5563; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; }

/* ── Skeleton loaders — MUST NOT BE ALTERED ── */
@keyframes skeletonPulse {
    0%   { opacity: 0.4; }
    50%  { opacity: 0.7; }
    100% { opacity: 0.4; }
}
.skeleton-block {
    background: #1E2329;
    border-radius: 12px;
    animation: skeletonPulse 1.6s ease-in-out infinite;
}
.skeleton-ring {
    background: #1E2329;
    border-radius: 50%;
    animation: skeletonPulse 1.6s ease-in-out infinite;
}
.skeleton-text {
    background: #262B34;
    border-radius: 6px;
    height: 12px;
    animation: skeletonPulse 1.6s ease-in-out infinite;
    margin: 4px 0;
}

/* ── Chart stability — MUST NOT BE ALTERED ── */
div[data-testid="stPlotlyChart"] { min-height: 195px; }
div[data-testid="column"]        { padding: 0 5px !important; }

/* ── Connection card ── */
.conn-card {
    background: #1E2329;
    border: 1px solid #2A2F3A;
    border-radius: 20px;
    padding: 60px 40px;
    text-align: center;
    margin: 40px auto;
    max-width: 500px;
}

/* ── Research banner ── */
.research-banner {
    background: linear-gradient(90deg, rgba(0,179,204,0.04) 0%, transparent 100%);
    border-left: 2px solid #00B3CC;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 14px;
}

/* ── History table ── */
.hist-row {
    display: flex; align-items: center;
    padding: 9px 14px; border-bottom: 1px solid #1A1D24;
    gap: 12px; font-size: 12px;
}
.hist-row:last-child { border-bottom: none; }
.hist-ts { color: #4B5563; font-family: 'DM Mono', monospace; font-size: 10px; min-width: 130px; }

.metric-table {
    width: 100%;
    font-size: 11px;
    color: #6B7280;
    font-family: 'DM Mono', monospace;
    border-collapse: collapse;
}
.metric-table td { padding: 4px 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. HISTORY LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class HistoryLogger:
    """
    Append-only JSON Lines session log.

    Record schema
    -------------
    ts            : ISO-8601 UTC timestamp
    subject       : int   (dataset subject index; -1 = live mode)
    mean_wl       : float (session average cognitive WORKLOAD 0–100)
    mean_fat      : float (session average fatigue index 0–100)
    accuracy      : float (CW accuracy 0–100; NaN for live)
    focus_minutes : float (cumulative focus streak minutes)
    """
    def __init__(self, path: Path = HISTORY_FILE):
        self.path = path

    def append(self, record: dict) -> None:
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def load_recent(self, n: int = 20) -> list:
        if not self.path.exists():
            return []
        lines   = self.path.read_text(encoding="utf-8").strip().splitlines()
        records = []
        for ln in reversed(lines[-n * 2:]):
            try:
                records.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return records[:n]


HISTORY = HistoryLogger()


# ══════════════════════════════════════════════════════════════════════════════
# 3. USER PROFILE SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def default_profile() -> dict:
    """Population-level prior from UNIVERSE dataset. count=1 prevents divide-by-zero."""
    return {"mean": 0.5, "M2": 0.0, "count": 1,
            "range_min": 0.35, "range_max": 0.65, "sessions": 0}

def load_profile(uid: str) -> dict:
    p = PROFILES_DIR / f"{uid}.json"
    return json.load(open(p)) if p.exists() else default_profile()

def save_profile(uid: str, profile: dict) -> None:
    json.dump(profile, open(PROFILES_DIR / f"{uid}.json", "w"), indent=2)

def list_users() -> list:
    return [p.stem for p in sorted(PROFILES_DIR.glob("*.json"))]


# ══════════════════════════════════════════════════════════════════════════════
# 4. PERSONALIZATION MATH  (Live mode only)
# ══════════════════════════════════════════════════════════════════════════════

def update_profile(profile: dict, cw_prob: float) -> dict:
    """Welford (1962) online mean/variance. Never called during Demo playback."""
    profile["count"] += 1
    delta = cw_prob - profile["mean"]
    profile["mean"] += delta / profile["count"]
    profile["M2"]   += delta * (cw_prob - profile["mean"])
    a = 0.02
    if cw_prob < profile["range_min"]:
        profile["range_min"] = (1 - a) * profile["range_min"] + a * cw_prob
    if cw_prob > profile["range_max"]:
        profile["range_max"] = (1 - a) * profile["range_max"] + a * cw_prob
    return profile

def personalized_std(p: dict) -> float:
    return float(np.sqrt(p["M2"] / p["count"])) if p["count"] >= 2 else 0.1

def personalized_threshold(p: dict) -> float:
    """threshold = μ + 0.5σ, clamped to [0.30, 0.80]."""
    return float(np.clip(p["mean"] + 0.5 * personalized_std(p), 0.30, 0.80))

def personalized_workload(cw_prob: float, p: dict) -> float:
    """
    Map raw cw_prob to 0–100 workload using user's personal CW range.

    Workload = ((cw_prob − r_min) / (r_max − r_min)) × 100

    High output = high workload = BAD.
    Normalises to the individual's personal range so the full 0–100 scale
    is used for every person regardless of their resting baseline.
    Contrast with personalized_bandwidth (v4.1) which was inverted:
    that mapped high P(CW) to low score. We now want high P(CW) → high %.
    """
    lo, hi = p["range_min"], p["range_max"]
    return round(np.clip((cw_prob - lo) / max(hi - lo, 0.05), 0, 1) * 100.0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 5. FOCUS STREAK
# ══════════════════════════════════════════════════════════════════════════════

class FocusStreak:
    """
    Tracks continuous low-CW windows as a focus streak.

    Grace period: streak resets only after GRACE_WINDOWS consecutive
    high-CW frames — single noisy frames do not break a valid streak.
    Inspired by heart-rate zone tracking (Garmin, 2019).

    Attributes
    ----------
    seconds       : total focused seconds this session
    grace_count   : consecutive high-CW frames since last focus window
    GRACE_WINDOWS : highs required to reset (default 3)
    WINDOW_SEC    : epoch duration in seconds (default 2)
    """
    GRACE_WINDOWS = 3
    WINDOW_SEC    = 2

    def __init__(self):
        self.seconds     = 0.0
        self.grace_count = 0

    def update(self, is_focused: bool) -> None:
        if is_focused:
            self.seconds    += self.WINDOW_SEC
            self.grace_count = 0
        else:
            self.grace_count += 1
            if self.grace_count >= self.GRACE_WINDOWS:
                self.seconds = 0.0; self.grace_count = 0

    @property
    def display(self) -> str:
        m, s = divmod(int(self.seconds), 60)
        return f"{m:02d}:{s:02d}"

    @property
    def minutes(self) -> float:
        return self.seconds / 60.0

    def reset(self) -> None:
        self.seconds = 0.0; self.grace_count = 0


# ══════════════════════════════════════════════════════════════════════════════
# 6. SIGNAL STABILITY GUARD
# ══════════════════════════════════════════════════════════════════════════════

def is_signal_stable(eeg_epoch: np.ndarray) -> Tuple[bool, str]:
    """
    Per-channel std heuristic after instance normalisation (std≈1.0 for clean signal).

    Rules
    -----
    std < 0.01 → flat line / disconnected electrode
    std > 3.0  → jaw clench / movement artefact

    Returns (stable: bool, reason: str).
    Reference: Viola et al. (2009) J Neurosci Methods 182(1):15-26.
    """
    ch_std = eeg_epoch.std(axis=1)
    if np.any(ch_std < 0.01):
        names = ["TP9", "AF7", "AF8", "TP10"]
        bad   = [names[i] for i in np.where(ch_std < 0.01)[0]]
        return False, f"Flat: {bad}"
    if np.any(ch_std > 3.0):
        return False, "Movement artefact"
    return True, "OK"


# ══════════════════════════════════════════════════════════════════════════════
# 7. MUSE STREAMER
# ══════════════════════════════════════════════════════════════════════════════

class MuseStreamer:
    """Non-blocking Muse EEG via pylsl daemon thread. Local Windows only."""
    SFREQ = 256
    WIN   = 512

    def __init__(self):
        self._buf     = collections.deque(maxlen=self.WIN * 4)
        self._inlet   = None
        self._thread  = None
        self._running = False

    def connect(self, timeout: float = 5.0) -> Tuple[bool, str]:
        if not PYLSL_AVAILABLE:
            return False, "pylsl not found. Run: python -m streamlit run dashboard.py"
        streams = pylsl.resolve_byprop("type", "EEG", timeout=timeout)
        if not streams:
            return False, "No Muse stream. Open BlueMuse → Start Streaming first."
        self._inlet   = pylsl.StreamInlet(streams[0])
        self._running = True
        self._thread  = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()
        return True, "Connected ✅"

    def _collect(self):
        while self._running and self._inlet:
            sample, _ = self._inlet.pull_sample(timeout=0.1)
            if sample:
                self._buf.append(np.array(sample[:4], dtype=np.float32) * 1e-6)

    def get_epoch(self) -> Optional[np.ndarray]:
        if len(self._buf) < self.WIN:
            return None
        return np.array(list(self._buf))[-self.WIN:].T.astype(np.float32)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 8. MODEL ARCHITECTURE  (must match BrainBatterySANN training checkpoint)
# ══════════════════════════════════════════════════════════════════════════════
SQI_DIM    = 6
TOWER_DIM  = 32
FUSION_DIM = 96


class SQIGate(nn.Module):
    """Maps SQI vector → per-modality quality weights ∈ (0,1)."""
    def __init__(self, sqi_dim: int = SQI_DIM):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(sqi_dim, 16), nn.ReLU(),
                                   nn.Linear(16, 3), nn.Sigmoid())
    def forward(self, sqi): return self.proj(sqi)


class AttentionFusion(nn.Module):
    """SQI-conditioned attention: att = softmax(content × (quality+0.1) / 3)."""
    def __init__(self, in_dim: int = FUSION_DIM, sqi_dim: int = SQI_DIM):
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
    """Lawhern et al. (2018) EEGNet + SE block. Input (B,1,4,512) → (B,F2)."""
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
    """1D CNN over BVP/HR/EDA. self.net matches physio_tower.net.* checkpoint keys."""
    def __init__(self, out_dim: int = TOWER_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 8, 15, padding=7), nn.BatchNorm1d(8), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(8, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(16, out_dim); self.drop = nn.Dropout(0.3)
    def forward(self, x): return self.fc(self.drop(self.net(x).squeeze(-1)))


class FreqEncoder(nn.Module):
    """MLP over 36 spectral features. self.net matches freq_tower.net.* checkpoint keys."""
    def __init__(self, in_dim: int = 36, out_dim: int = TOWER_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.BatchNorm1d(64),
                                  nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))
    def forward(self, x): return self.net(x)


class MultimodalCWSANN(nn.Module):
    """
    Subject-Agnostic Neural Network — cognitive workload classification.
    Inference-only. Predicts P(high CW) — NOT emotional stress.
    Expected missing keys (harmless): stress_head, subject_head, projector.
    """
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
# 9. RUNTIME CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H5_PATH = DATA_DIR
PT_PATH = MODEL_PATH


# ══════════════════════════════════════════════════════════════════════════════
# 10. CACHED LOADERS
# CRITICAL: Zero st.* calls inside these functions.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(path: str) -> Tuple[Optional[nn.Module], str, dict]:
    """Returns (model|None, status_str, diag_dict). Zero st.* calls."""
    diag = {}
    if not os.path.exists(path):
        return None, f"File not found: {path}", diag
    model = MultimodalCWSANN()
    ckpt  = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    diag["ckpt_keys"] = str(list(ckpt.keys())[:6])
    result   = model.load_state_dict(ckpt, strict=False)
    expected = {"stress_head", "subject_head", "projector"}
    critical = [k for k in result.missing_keys if not any(x in k for x in expected)]
    if critical:
        diag["status"] = f"❌ Missing: {critical[:4]}"
        return None, "Architecture mismatch", diag
    diag["status"] = "✅ All inference keys loaded"
    model = model.to(DEVICE).eval()
    try:
        with torch.inference_mode():
            out, _ = model(torch.zeros(1,1,4,512,device=DEVICE),
                           torch.zeros(1,3,512,device=DEVICE),
                           torch.zeros(1,36,device=DEVICE))
        diag["fwd"] = f"✅ Forward pass OK {tuple(out.shape)}"
    except Exception as e:
        return None, f"Forward pass failed: {e}", diag
    return model, "OK", diag


@st.cache_data(show_spinner=False)
def load_subject_epochs(
    data_dir: str, subject_idx: int, num_epochs: int = 400,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Load per-subject epochs from .npy files with memory-mapping.
    Balanced class selection: most-recent N epochs from each class.
    Instance normalisation applied here (Schirrmeister et al., 2017).
    """
    data_dir = Path(data_dir)
    for fname in ["eeg.npy", "physio.npy", "label_workload.npy", "subjects.npy"]:
        if not (data_dir / fname).exists():
            return None, None, None, f"Missing file: {data_dir / fname}"

    eeg_all      = np.load(data_dir / "eeg.npy",           mmap_mode="r")
    physio_all   = np.load(data_dir / "physio.npy",         mmap_mode="r")
    labels_all   = np.load(data_dir / "label_workload.npy", mmap_mode="r")
    subjects_all = np.load(data_dir / "subjects.npy",       mmap_mode="r")

    subj_mask = subjects_all == subject_idx
    if not np.any(subj_mask):
        return None, None, None, f"No epochs for subject {subject_idx}"

    subj_idx_global = np.where(subj_mask)[0]
    subj_labels     = labels_all[subj_idx_global]
    low_pos         = np.where(subj_labels == 0)[0]
    high_pos        = np.where(subj_labels == 1)[0]

    if len(low_pos) == 0 or len(high_pos) == 0:
        selected = subj_idx_global[:num_epochs]
    else:
        n_each   = min(len(low_pos), len(high_pos), num_epochs // 2)
        selected = np.sort(np.concatenate([
            subj_idx_global[low_pos[-n_each:]], subj_idx_global[high_pos[:n_each]]]))

    eeg    = np.array(eeg_all[selected],    dtype=np.float32)
    physio = np.array(physio_all[selected], dtype=np.float32)
    labels = np.array(labels_all[selected], dtype=np.int8)

    for arr in (eeg, physio):
        mu  = arr.mean(axis=2, keepdims=True)
        sig = arr.std(axis=2,  keepdims=True) + 1e-8
        arr[:] = np.clip((arr - mu) / sig, -5.0, 5.0)

    return eeg, physio, labels, "OK"


@st.cache_data(hash_funcs={np.ndarray: lambda x: (x.shape, int(x.sum()))})
def precompute_freq_features(eeg_all: np.ndarray, physio_all: np.ndarray) -> np.ndarray:
    """
    36-dim spectral feature vectors matching FREQ_IN_DIM=36 from training.
    [0:20]  log band power per ch (δ θ α β γ × 4 ch)
    [20:24] log β/(θ+α) per ch — CW marker (Holm et al., 2009)
    [24:28] log θ/α per ch — fatigue proxy (Klimesch, 1999)
    [28:33] frontal asymmetry log(AF8/AF7) per band
    [33:36] HR mean, HR std, log-EDA mean
    """
    N = eeg_all.shape[0]; feats = np.zeros((N, 36), dtype=np.float32)
    bands = [(1,4),(4,8),(8,13),(13,30),(30,40)]
    for i in range(N):
        freqs, psd = welch(eeg_all[i], fs=256.0, nperseg=256, axis=1)
        bp      = [np.clip(np.mean(psd[:,(freqs>=lo)&(freqs<hi)],axis=1),1e-12,1e3)
                   for lo,hi in bands]
        psd_mat = np.stack(bp, axis=1)
        b_r     = np.log(psd_mat[:,3]/(psd_mat[:,1]+psd_mat[:,2]+1e-12))
        t_r     = np.log(psd_mat[:,1]/(psd_mat[:,2]+1e-12))
        asym    = np.log(psd_mat[2,:]+1e-12) - np.log(psd_mat[1,:]+1e-12)
        hr, eda = physio_all[i,1,:], physio_all[i,2,:]
        stats   = np.array([np.mean(hr), np.std(hr),
                             np.log(np.mean(eda)-np.min(eda)+1e-8)], dtype=np.float32)
        feats[i] = np.concatenate([np.log(psd_mat).flatten(), b_r, t_r, asym, stats])
    return feats


# ══════════════════════════════════════════════════════════════════════════════
# 11. SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _reset_history():
    """
    Clear display history and accuracy counters.
    wl_hist initialised to 0.0 (0% workload = resting state).
    fatigue_hist initialised to 0.0.
    Called on: subject change, mode switch, restart button.
    """
    st.session_state.wl_hist      = collections.deque([0.0]*100, maxlen=100)
    st.session_state.fatigue_hist = collections.deque([0.0]*100, maxlen=100)
    st.session_state.time_hist    = collections.deque(range(-100,0), maxlen=100)
    st.session_state.n_correct    = 0
    st.session_state.n_total      = 0
    st.session_state.fss          = 0
    st.session_state._run_complete = False
    st.session_state._final_stats  = {}
    if "streak" in st.session_state:
        st.session_state.streak.reset()


_defaults: dict = {
    "running":          False,
    "wl_hist":          collections.deque([0.0]*100, maxlen=100),
    "fatigue_hist":     collections.deque([0.0]*100, maxlen=100),
    "time_hist":        collections.deque(range(-100,0), maxlen=100),
    "n_correct":        0, "n_total": 0,
    "subject_idx":      11, "frame_idx": 0,
    "user_id":          "default",
    "profile":          default_profile(),
    "fss":              0,
    "streak":           FocusStreak(),
    "app_mode":         "📼 Demo — pre-recorded subjects",
    "active_tab":       "⚡ Live",
    "muse_streamer":    None, "muse_connected": False,
    "signal_stable":    True, "signal_reason": "OK", "stable_count": 0,
    "speed":            0.5, "ema_alpha": 0.7,
    "show_gt":          True, "lite_mode": False,
    "_model":           None,
    "_eeg_data":        None, "_physio_data": None,
    "_labels_data":     None, "_freq_all":    None,
    "_run_complete":    False, "_final_stats": {},
    "_loaded_subject":  -1,
    # Debug — written by fragment, read by sidebar OUTSIDE the fragment
    "_dbg_cw":          0.0, "_dbg_frame": 0,
    "_dbg_freq_mean":   0.0, "_dbg_freq_std": 0.0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# 12. COLOUR HELPERS + CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _wl_color(wl: float) -> str:
    """
    Workload colour.  High workload = BAD, so palette is inverted vs bandwidth:
      0–30%  → green  (LOW EFFORT)
      30–70% → amber  (ACTIVE)
      70–100%→ red    (HEAVY LOAD)
    """
    return "#00CC77" if wl < 30 else "#E09000" if wl < 70 else "#E05050"

def _fat_color(fat: float) -> str:
    return "#00CC77" if fat < 35 else "#E09000" if fat < 65 else "#E05050"

def _wl_pill_class(wl: float) -> str:
    return "pill-green" if wl < 30 else "pill-orange" if wl < 70 else "pill-red"

def _wl_pill(wl: float) -> str:
    """Status pill for cognitive workload level."""
    lbl = "LOW EFFORT" if wl < 30 else "ACTIVE" if wl < 70 else "HEAVY LOAD"
    return f'<span class="pill {_wl_pill_class(wl)}">{lbl}</span>'


# ── Skeleton helpers — DO NOT ALTER ──────────────────────────────────────────
def _skel_bar(h: int = 200) -> str:
    return (f'<div class="skeleton-block" '
            f'style="height:{h}px;width:100%;border-radius:12px"></div>')

def _skel_ring() -> str:
    return ('<div style="display:flex;justify-content:center;align-items:center;height:195px">'
            '<div class="skeleton-ring" style="width:160px;height:160px"></div></div>')

def _skel_text_block() -> str:
    return (''.join(
        f'<div class="skeleton-text" style="width:{w}%;margin-bottom:8px"></div>'
        for w in [70, 90, 60, 80, 55]
    ))


def make_ring(value: float, color: str, label: str, subtitle: str = "") -> go.Figure:
    """Donut ring gauge. value 0–100."""
    fig = go.Figure()
    fig.add_trace(go.Pie(values=[100], hole=0.72, marker_colors=["#262B34"],
                          textinfo="none", hoverinfo="skip", showlegend=False))
    v = max(float(value), 0.5)
    fig.add_trace(go.Pie(values=[v, 100-v], hole=0.72,
                          marker_colors=[color, "rgba(0,0,0,0)"],
                          textinfo="none", hoverinfo="skip",
                          showlegend=False, sort=False, rotation=90, direction="clockwise"))
    inner = f"<b>{value:.0f}</b><br><span style='font-size:10px;color:#4B5563'>{label}</span>"
    if subtitle:
        inner += f"<br><span style='font-size:9px;color:#374151'>{subtitle}</span>"
    fig.add_annotation(text=inner, x=0.5, y=0.5, showarrow=False,
                        font=dict(color=color, family="DM Mono", size=30), align="center")
    fig.update_layout(height=195, margin=dict(l=8,r=8,t=8,b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False)
    return fig


def make_workload_bar_html(wl: float, gt_str: str = "") -> str:
    """
    Horizontal energy bar for the hero cognitive workload metric.

    The bar fills left-to-right as workload increases (opposite of bandwidth).
    0% (left, "Resting") = brain at rest.
    100% (right, "Overloaded") = maximum cognitive demand.

    Color uses _wl_color: green → amber → red as load rises.
    """
    pct   = max(min(wl, 100), 0)
    color = _wl_color(wl)
    grad  = f"linear-gradient(90deg, {color}55 0%, {color} 100%)"
    sub   = (f'<span style="color:#4B5563;font-size:11px;'
             f'font-family:\'DM Mono\',monospace">&nbsp;{gt_str}</span>') if gt_str else ""
    return f"""
    <div class="bb-card-hero" style="min-height:110px">
      <span class="bb-label">Cognitive Workload</span>
      <div style="display:flex;align-items:baseline;gap:10px;margin:6px 0 4px 0">
        <span class="bb-value" style="color:{color};font-size:52px">{pct:.0f}</span>
        <span class="bb-sub" style="font-size:16px">/ 100%</span>
        {_wl_pill(wl)}{sub}
      </div>
      <div class="energy-bar-wrap">
        <div class="energy-bar-fill" style="width:{pct}%;background:{grad}"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:4px">
        <span class="bb-sub">Resting</span>
        <span class="bb-sub">Overloaded</span>
      </div>
    </div>"""


def make_trend_chart(time_x: list, wl_hist: list,
                     fatigue_hist: list, p_threshold: float) -> go.Figure:
    """
    60-second rolling trend: Workload + Fatigue + threshold line.
    Threshold line drawn at p_threshold × 100 (workload space).
    High values = heavy load = bad.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_x, y=wl_hist, mode="lines", name="Cognitive Workload",
        line=dict(color="#00B3CC", width=2.2),
        fill="tozeroy", fillcolor="rgba(0,179,204,0.05)"))
    fig.add_trace(go.Scatter(
        x=time_x, y=fatigue_hist, mode="lines", name="Fatigue (θ/α)",
        line=dict(color="#A06EDC", width=1.5, dash="dot")))
    # Threshold in workload space = p_threshold × 100
    fig.add_hline(y=p_threshold * 100.0, line_dash="dash",
                   line_color="rgba(224,144,0,0.4)",
                   annotation_text="CW threshold",
                   annotation_font=dict(color="#E09000", size=10))
    fig.update_layout(
        height=200, margin=dict(l=8,r=8,t=8,b=28),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0,100], gridcolor="#1A1D24", color="#4B5563", tickfont=dict(size=9)),
        xaxis=dict(showgrid=False, color="#4B5563", tickfont=dict(size=9)),
        legend=dict(x=0.01, y=0.99, font=dict(color="#8B949E", size=9), bgcolor="rgba(0,0,0,0)"),
        uirevision="keep")
    return fig


def make_attention_chart(attn_vals: np.ndarray) -> go.Figure:
    """Bar chart showing which sensor tower the model is relying on most."""
    fig = go.Figure(go.Bar(
        x=["EEG", "Physio", "Spectral"], y=[float(v) for v in attn_vals],
        marker_color=["#00B3CC", "#E05050", "#A06EDC"],
        marker_line_width=0,
        text=[f"{v:.2f}" for v in attn_vals],
        textposition="outside",
        textfont=dict(color="#8B949E", size=10, family="DM Mono")))
    fig.update_layout(
        height=195, margin=dict(l=5,r=5,t=16,b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0,1.35], showgrid=False, visible=False),
        xaxis=dict(color="#6B7280", tickfont=dict(size=10, family="DM Mono")),
        bargap=0.35)
    return fig


def make_eeg_chart(eeg_ep: np.ndarray) -> go.Figure:
    ds = slice(None, None, 2)
    fig = go.Figure()
    for j,(color,ch) in enumerate(zip(
            ["#00B3CC","#A06EDC","#00CC77","#E09000"], ["TP9","AF7","AF8","TP10"])):
        fig.add_trace(go.Scatter(y=eeg_ep[j,ds]*8.0+j*7, mode="lines", name=ch,
                                  line=dict(color=color, width=1.0)))
    fig.update_layout(height=180, margin=dict(l=5,r=5,t=5,b=5),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_visible=False, yaxis=dict(range=[-10,32], visible=False),
                      legend=dict(font=dict(color="#6B7280",size=9),
                                  bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.05))
    return fig


def make_physio_chart(physio_ep: np.ndarray) -> go.Figure:
    ds = slice(None, None, 2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=physio_ep[0,ds], mode="lines", name="BVP",
                              line=dict(color="#E05050", width=1.4)))
    fig.add_trace(go.Scatter(y=physio_ep[2,ds]+3, mode="lines", name="EDA",
                              line=dict(color="#E09000", width=1.8)))
    fig.add_trace(go.Scatter(y=physio_ep[1,ds]-3, mode="lines", name="HR",
                              line=dict(color="#00CC77", width=1.4)))
    fig.update_layout(height=180, margin=dict(l=5,r=5,t=5,b=5),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_visible=False, yaxis_visible=False,
                      legend=dict(font=dict(color="#6B7280",size=9),
                                  bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.05))
    return fig


def make_history_chart(records: list) -> go.Figure:
    if not records:
        fig = go.Figure()
        fig.add_annotation(text="No sessions recorded yet", x=0.5, y=0.5,
                            showarrow=False, font=dict(color="#4B5563", size=14))
        fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        return fig
    recs    = list(reversed(records[:7]))
    labels  = [r.get("ts","")[:10] for r in recs]
    # Support both legacy "mean_bw" key and new "mean_wl" key
    wl_vals = [r.get("mean_wl", r.get("mean_bw", 0)) for r in recs]
    fig = go.Figure(go.Bar(
        x=labels, y=wl_vals,
        marker_color=[_wl_color(v) for v in wl_vals], marker_line_width=0,
        text=[f"{v:.0f}%" for v in wl_vals], textposition="outside",
        textfont=dict(color="#8B949E", size=10, family="DM Mono")))
    # Reference line at 70% (heavy-load boundary)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(224,80,80,0.25)")
    fig.update_layout(height=220, margin=dict(l=5,r=5,t=20,b=5),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      yaxis=dict(range=[0,115], showgrid=False, visible=False),
                      xaxis=dict(color="#6B7280", tickfont=dict(size=9, family="DM Mono")),
                      bargap=0.25)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 13. SIDEBAR
# NOTE: st.sidebar.* FORBIDDEN inside @st.fragment.
# Fragment writes debug to st.session_state._dbg_*, sidebar reads here.
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="bb-title">⚙ Controls</p>', unsafe_allow_html=True)
    st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

    app_mode = st.radio("Input source",
                         ["📼 Demo — pre-recorded subjects", "🎧 Live — Muse headband"],
                         index=["📼 Demo — pre-recorded subjects",
                                "🎧 Live — Muse headband"].index(st.session_state.app_mode))
    if app_mode != st.session_state.app_mode:
        st.session_state.app_mode        = app_mode
        st.session_state.running         = False
        st.session_state.muse_connected  = False
        st.session_state._eeg_data       = None
        st.session_state._loaded_subject = -1
        _reset_history()
        st.rerun()

    st.markdown("---")
    st.markdown('<span class="bb-label">👤 User profile</span>', unsafe_allow_html=True)
    existing = list_users()
    options  = (existing or []) + ["➕ New user…"]
    safe_idx = existing.index(st.session_state.user_id) if st.session_state.user_id in existing else 0
    sel_user = st.selectbox("Who is using the system?", options, index=safe_idx)

    if sel_user == "➕ New user…":
        new_name = st.text_input("Name (letters/numbers)")
        if st.button("Create") and new_name.strip():
            clean = "".join(c for c in new_name.strip() if c.isalnum() or c == "_")
            if clean:
                save_profile(clean, default_profile())
                st.session_state.user_id = clean
                st.session_state.profile = default_profile()
                _reset_history()
                st.rerun()
    else:
        if sel_user != st.session_state.user_id:
            st.session_state.user_id = sel_user
            st.session_state.profile = load_profile(sel_user)
            _reset_history()
            st.rerun()

    profile = st.session_state.profile
    obs     = profile["count"] - 1

    if app_mode == "🎧 Live — Muse headband":
        st.progress(min(obs/200.0, 1.0), text=f"Personalizing… {obs}/200")
        if obs < 50:    st.info("🔵 Zero-calibration (population prior)")
        elif obs < 200: st.warning(f"🟡 Calibrating ({obs}/200)")
        else:           st.success("🟢 Personalized")
        st.caption(f"Mean CW: {profile['mean']:.3f}  |  "
                   f"Threshold: {personalized_threshold(profile):.3f}")
    else:
        st.caption("Personalization active in Live mode only.")

    if st.button("Reset my profile"):
        st.session_state.profile = default_profile()
        save_profile(st.session_state.user_id, default_profile())
        _reset_history()
        st.rerun()

    st.markdown("---")
    if app_mode == "📼 Demo — pre-recorded subjects":
        st.markdown('<span class="bb-label">📊 Dataset subject</span>', unsafe_allow_html=True)
        subj_sel = st.selectbox(
            "Replay subject",
            list(range(24)),
            format_func=lambda x: f"Subject {x + 1}",
            index=min(st.session_state.subject_idx, 23),
        )
        # ── TASK 1 FIX: explicit running=False + _reset_history() before rerun ──
        if subj_sel != st.session_state.subject_idx:
            st.session_state.subject_idx     = subj_sel
            st.session_state.frame_idx       = 0
            st.session_state.running         = False   # ← explicit: kill live loop
            st.session_state._eeg_data       = None
            st.session_state._physio_data    = None
            st.session_state._labels_data    = None
            st.session_state._freq_all       = None
            st.session_state._loaded_subject = -1
            _reset_history()                           # ← wipes wl_hist, counters
            st.rerun()
        st.caption("Arc: last 200 calm → first 200 high-load epochs")
        st.markdown("---")

    st.markdown('<span class="bb-label">⚙ Playback</span>', unsafe_allow_html=True)
    st.session_state.speed     = st.slider("Frame delay (s)", 0.1, 2.0, 0.5, 0.1)
    st.session_state.ema_alpha = st.slider("Smoothing (α)",   0.1, 1.0, 0.7, 0.05,
                                            help="display[t] = α·model[t] + (1−α)·display[t−1]")
    st.session_state.show_gt   = st.checkbox("Show ground truth label", value=True)
    st.session_state.lite_mode = st.checkbox("Low-bandwidth mode", value=False,
                                              help="Hides raw waveforms.")
    st.markdown("---")
    st.caption(f"Device: {DEVICE}")

    # Debug — written by fragment, read here (safe: outside fragment)
    if st.session_state.get("_dbg_frame", 0) > 0:
        st.markdown('<span class="bb-label">🔬 Debug</span>', unsafe_allow_html=True)
        st.caption(
            f"Frame {st.session_state._dbg_frame}  "
            f"P(CW)={st.session_state._dbg_cw:.3f}\n"
            f"freq μ={st.session_state._dbg_freq_mean:.3f}  "
            f"σ={st.session_state._dbg_freq_std:.3f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 14. LOAD MODEL + DATA  (outside fragment — safe to call st.sidebar here)
# ══════════════════════════════════════════════════════════════════════════════
model, model_status, model_diag = load_model(PT_PATH)

with st.sidebar:
    st.markdown("---")
    st.markdown('<span class="bb-label">🤖 Model status</span>', unsafe_allow_html=True)
    ds_label = model_diag.get("status", "")
    if "❌" in ds_label: st.error(ds_label)
    elif ds_label:       st.success(ds_label)
    if model_diag.get("fwd"): st.success(model_diag["fwd"])
    st.caption(f"Keys (first 6): {model_diag.get('ckpt_keys','N/A')}")

st.session_state._model = model
is_demo = st.session_state.app_mode == "📼 Demo — pre-recorded subjects"

if is_demo:
    need_load = (st.session_state._eeg_data is None
                 or st.session_state._loaded_subject != st.session_state.subject_idx)
    if need_load:
        eeg_d, physio_d, labels_d, data_status = load_subject_epochs(
            H5_PATH, st.session_state.subject_idx)
        if data_status == "OK":
            freq_d = precompute_freq_features(eeg_d, physio_d)
            st.session_state._eeg_data       = eeg_d
            st.session_state._physio_data    = physio_d
            st.session_state._labels_data    = labels_d
            st.session_state._freq_all       = freq_d
            st.session_state._loaded_subject = st.session_state.subject_idx
        else:
            st.error(f"❌ Data: {data_status}"); st.stop()
    if model_status != "OK":
        st.error(f"❌ Model: {model_status}"); st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 15. STATIC PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
hc1, hc2 = st.columns([3, 1])
with hc1:
    st.markdown(
        '<h1 style="margin:0;font-size:26px;color:#00B3CC;'
        'font-family:\'DM Mono\',monospace">🧠 Brain Battery</h1>',
        unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#4B5563;font-size:13px;margin-top:2px">'
        'Cognitive Workload Monitor — UNIVERSE Dataset (n=12 subjects, LOSO)</p>',
        unsafe_allow_html=True)
with hc2:
    if is_demo and st.session_state._labels_data is not None:
        n_hi = int(np.sum(st.session_state._labels_data == 1))
        n_lo = int(np.sum(st.session_state._labels_data == 0))
        st.markdown(
            f'<div style="text-align:right;margin-top:8px">'
            f'<span class="pill pill-dim">Subject {st.session_state.subject_idx+1}</span>&nbsp;'
            f'<span class="pill pill-blue">{n_lo}↓ {n_hi}↑</span></div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="research-banner">
  <span style="color:#374151;font-size:10px;font-family:'DM Mono',monospace;
               text-transform:uppercase;letter-spacing:0.12em">Core Research Claim</span><br>
  <span style="color:#8B949E;font-size:13px;font-family:'DM Sans',sans-serif">
    Spectral features generalize across subjects better than raw EEG or physiological signals —
    but no modality achieves reliable zero-calibration at the individual level.
  </span>
</div>
""", unsafe_allow_html=True)

TABS = ["⚡ Live", "🏠 Summary", "📂 History"]
tc   = st.columns(len(TABS))
for i, tab in enumerate(TABS):
    with tc[i]:
        if st.button(tab, key=f"tab_{i}", use_container_width=True,
                     type="primary" if st.session_state.active_tab == tab else "secondary"):
            st.session_state.active_tab = tab; st.rerun()

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 16. LIVE MODE GATE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.app_mode == "🎧 Live — Muse headband":
    if not PYLSL_AVAILABLE:
        st.markdown("""
        <div class="conn-card">
          <div style="font-size:52px;margin-bottom:14px">📡</div>
          <h3 style="color:#E05050;margin:0 0 8px 0;font-family:'DM Mono',monospace">
            pylsl Not Found</h3>
          <p style="color:#6B7280;font-size:13px;margin-bottom:16px">
            Streamlit is using a different Python than where pylsl is installed.</p>
          <code style="background:#0A0C10;padding:10px 18px;border-radius:8px;
                       color:#00B3CC;font-family:'DM Mono',monospace;font-size:13px">
            python -m streamlit run dashboard.py</code>
        </div>""", unsafe_allow_html=True)
        st.stop()

    if not st.session_state.muse_connected:
        st.markdown("""
        <div class="conn-card">
          <div style="font-size:52px;margin-bottom:14px">🎧</div>
          <h3 style="color:#00B3CC;margin:0 0 8px 0;font-family:'DM Mono',monospace">
            Waiting for Muse Headband</h3>
          <p style="color:#6B7280;font-size:13px;white-space:pre-line;margin-bottom:20px">
1. Put on the Muse headband
2. Open BlueMuse → Start Streaming
3. Click Connect below</p>
        </div>""", unsafe_allow_html=True)
        col_btn = st.columns([2,1,2])[1]
        with col_btn:
            if st.button("🔌 Connect", type="primary", use_container_width=True):
                streamer = MuseStreamer()
                ok, msg  = streamer.connect(timeout=5.0)
                if ok:
                    st.session_state.muse_streamer  = streamer
                    st.session_state.muse_connected = True
                    st.success(msg); st.rerun()
                else:
                    st.error(msg)
        st.stop()
    else:
        stable = st.session_state.get("signal_stable", True)
        reason = st.session_state.get("signal_reason", "OK")
        st.success("🎧 Muse connected — signal stable") if stable else st.warning(f"⚠️ {reason}")


# ══════════════════════════════════════════════════════════════════════════════
# 17. HISTORY TAB
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "📂 History":
    records = HISTORY.load_recent(20)
    st.markdown('<p class="bb-label">Session History (most recent first)</p>',
                unsafe_allow_html=True)
    hcol1, hcol2 = st.columns([2, 1])
    with hcol1:
        st.plotly_chart(make_history_chart(records), use_container_width=True, key="hist_chart")
    with hcol2:
        if records:
            st.markdown('<div class="bb-card">', unsafe_allow_html=True)
            for r in records[:8]:
                wl     = r.get("mean_wl", r.get("mean_bw", 0))
                col    = _wl_color(wl)
                ts     = r.get("ts","")[:16].replace("T"," ")
                subj   = r.get("subject", -1)
                subj_s = f"S{subj+1}" if subj >= 0 else "Live"
                acc    = r.get("accuracy", float("nan"))
                acc_s  = f"{acc:.0f}%" if not (isinstance(acc, float) and np.isnan(acc)) else "—"
                st.markdown(
                    f'<div class="hist-row">'
                    f'<span class="hist-ts">{ts}</span>'
                    f'<span style="color:#6B7280;font-size:11px">{subj_s}</span>'
                    f'<span style="color:{col};font-family:\'DM Mono\',monospace;font-size:12px">'
                    f'{wl:.0f}% WL</span>'
                    f'<span style="color:#6B7280;font-size:11px">{acc_s} acc</span>'
                    f'</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="bb-card" style="text-align:center;color:#4B5563;padding:40px">'
                'No sessions yet.<br>Run a demo to start logging.</div>',
                unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 18. SUMMARY TAB
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "🏠 Summary":
    records = HISTORY.load_recent(10)
    if records:
        mean_wl_all  = float(np.mean([r.get("mean_wl", r.get("mean_bw", 50)) for r in records]))
        mean_fat_all = float(np.mean([r.get("mean_fat", 50) for r in records]))
        total_focus  = sum(r.get("focus_minutes", 0) for r in records)
        n_sessions   = len(records)
    else:
        mean_wl_all = mean_fat_all = total_focus = 0.0; n_sessions = 0

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Sessions logged",    str(n_sessions))
    sc2.metric("Avg workload",       f"{mean_wl_all:.0f}%")
    sc3.metric("Avg fatigue index",  f"{mean_fat_all:.0f}/100")
    sc4.metric("Total focus time",   f"{total_focus:.1f} min")

    st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

    if is_demo and st.session_state._labels_data is not None:
        n_hi = int(np.sum(st.session_state._labels_data==1))
        n_lo = int(np.sum(st.session_state._labels_data==0))
        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("LOSO accuracy (avg)", "~50%",       "Best subject 53.5% (S12)")
        rm2.metric("Best signal",  "Freq bands",         "σ=0.031 across 12 subjects")
        rm3.metric("Worst case",   "Subject 2: 27.5%",   "Fails zero-calibration")
        rm4.metric("This window",  f"{n_lo}↓ + {n_hi}↑","Workload rises at midpoint")

    with st.expander("📖 What does the model measure?", expanded=False):
        st.markdown("""
**Cognitive Workload** — mental effort required by a task. Not emotional stress.
```
P(high CW) = softmax( workload_head( EEG ⊕ Physio ⊕ Freq ) )₁
Workload  = 100 × P(high CW)   ← 0% rested, 100% maxed out
Fatigue   = log(θ power) − log(α power)   [Klimesch 1999]
```
**Signal Weights** — which sensor the model relied on most.
Spectral features generalize best: consistent across all 12 subjects.

**What it does NOT measure:** emotional stress, HRV, cortisol.

**Demo moment to watch:**
Jaw clench → spectral weight drops → model automatically down-weights EEG.
> *"Most systems give you a number. Ours tells you when not to trust it."*
        """)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 19. LIVE FRAGMENT  — Vertical Health Feed
# ──────────────────────────────────────────────────────────────────────────────
# CRITICAL RULES (all retained from v4.1)
# 1. ZERO st.sidebar.* calls anywhere in this function.
# 2. ALL layout scaffolding declared UNCONDITIONALLY at the top.
# 3. Skeleton loaders fill all placeholders before inference starts.
#
# STATE MACHINE (v4.2 fix)
# ─────────────────────────
#   [IDLE] ──Start──▶ [RUNNING] ──End──▶ [COMPLETE]
#                          ▲                  │
#                          └──Restart──────────┘
#
# "↺ Restart Run" button: resets all deques, frame_idx, _run_complete,
# sets running=True, calls st.rerun() IMMEDIATELY (no fall-through).
#
# The loop driver at the bottom calls st.rerun() whenever running=True,
# regardless of whether data was acquired this tick — this prevents the
# fragment from silently dying after a subject switch or any other
# state transition that sets running=True without an explicit rerun.
# ══════════════════════════════════════════════════════════════════════════════

@st.fragment
def live_display():
    speed      = st.session_state.get("speed",     0.5)
    ema_alpha  = st.session_state.get("ema_alpha", 0.7)
    show_gt    = st.session_state.get("show_gt",   True)
    lite_mode  = st.session_state.get("lite_mode", False)
    frag_demo  = st.session_state.app_mode == "📼 Demo — pre-recorded subjects"
    frag_model = st.session_state.get("_model",       None)
    f_eeg      = st.session_state.get("_eeg_data",    None)
    f_physio   = st.session_state.get("_physio_data", None)
    f_labels   = st.session_state.get("_labels_data", None)
    f_freq     = st.session_state.get("_freq_all",    None)
    streak     = st.session_state.get("streak", FocusStreak())

    if frag_model is None:
        st.warning("⏳ Model not loaded — check sidebar diagnostics."); return
    frag_model.eval()

    # ── Run-complete summary (top of layout, above all rows) ──────────────
    if st.session_state._run_complete:
        fs       = st.session_state._final_stats
        acc_f    = fs.get("accuracy",      0.0)
        n_cor    = fs.get("n_correct",     0)
        n_tot    = fs.get("n_total",       0)
        mean_wl  = fs.get("mean_wl",       0.0)
        mean_fat = fs.get("mean_fat",      0.0)
        focus_m  = fs.get("focus_minutes", 0.0)
        subj_num = st.session_state.subject_idx + 1
        acc_col  = "#00CC77" if acc_f>=60 else "#E09000" if acc_f>=45 else "#E05050"
        wl_str   = ("brain at low effort" if mean_wl < 30
                    else "moderate workload throughout" if mean_wl < 70
                    else "sustained heavy cognitive demand")
        fat_str  = ("stayed alert" if mean_fat<35
                    else "some tiredness built up" if mean_fat<65 else "significant fatigue")

        st.markdown(f"""
        <div class="bb-card-accent" style="margin-bottom:14px">
          <h2 style="color:#00B3CC;margin-top:0;font-family:'DM Mono',monospace">
            ✅ Run Complete — Subject {subj_num}</h2>
          <table class="metric-table">
            <tr style="border-bottom:1px solid #1A1D24">
              <td style="padding:7px 0;color:#4B5563;font-size:10px;text-transform:uppercase;
                         letter-spacing:0.1em">Windows analysed</td>
              <td style="color:#E5E7EB"><b>{n_tot}</b>
                <span style="color:#4B5563;font-size:10px"> × 2 s</span></td></tr>
            <tr style="border-bottom:1px solid #1A1D24">
              <td style="padding:7px 0;color:#4B5563;font-size:10px;text-transform:uppercase;
                         letter-spacing:0.1em">Classification accuracy</td>
              <td style="color:{acc_col}"><b>{acc_f:.1f}%</b>
                <span style="color:#4B5563;font-size:10px">
                  &nbsp;({n_cor}/{n_tot} correct — chance 50%)</span></td></tr>
            <tr style="border-bottom:1px solid #1A1D24">
              <td style="padding:7px 0;color:#4B5563;font-size:10px;text-transform:uppercase;
                         letter-spacing:0.1em">Average workload</td>
              <td style="color:#00B3CC"><b>{mean_wl:.1f}%</b>
                <span style="color:#4B5563;font-size:10px"> → {wl_str}</span></td></tr>
            <tr style="border-bottom:1px solid #1A1D24">
              <td style="padding:7px 0;color:#4B5563;font-size:10px;text-transform:uppercase;
                         letter-spacing:0.1em">Average fatigue (θ/α)</td>
              <td style="color:#A06EDC"><b>{mean_fat:.1f}/100</b>
                <span style="color:#4B5563;font-size:10px"> → {fat_str}</span></td></tr>
            <tr>
              <td style="padding:7px 0;color:#4B5563;font-size:10px;text-transform:uppercase;
                         letter-spacing:0.1em">Focus streak time</td>
              <td style="color:#00CC77"><b>{focus_m:.1f} min</b></td></tr>
          </table>
          <p style="color:#4B5563;font-size:11px;margin:12px 0 0 0">
            Click "↺ Restart Run" below to replay, or choose a different subject.</p>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # LAYOUT SCAFFOLD — UNCONDITIONAL, before all state checks.
    # DO NOT MOVE OR CONDITIONALISE ANY OF THESE DECLARATIONS.
    # ══════════════════════════════════════════════════════════════════════

    # ── Row 1: Cognitive Workload (hero) ───────────────────────────────────
    r1l, r1r = st.columns([2, 1])
    with r1l:
        ph_hero = st.empty()
    with r1r:
        st.markdown(
            '<p class="feed-section-title">Cognitive Workload</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p class="bb-desc">'
            '0% = fully rested. 100% = brain maxed out. '
            'Measures how much mental effort your current task is demanding.</p>',
            unsafe_allow_html=True)
        with st.popover("🔬 Technical Details"):
            st.markdown("""
**How it's calculated**
```
P(high CW) = softmax(
    workload_head( EEG ⊕ Physio ⊕ Freq )
)₁
Workload = 100 × P(high CW)
```
Trained on NASA-TLX binary labels, UNIVERSE dataset (n=12 subjects, LOSO).
Threshold fixed at 0.5 in Demo mode; personalized via Welford algorithm in Live mode.

**What it is NOT:** emotional stress — that requires HRV, phasic EDA, cortisol.
            """)
        ph_wl_pill = st.empty()

    st.markdown('<div class="bb-divider" style="margin:6px 0"></div>', unsafe_allow_html=True)

    # ── Row 2: Mental Fatigue ──────────────────────────────────────────────
    r2l, r2r = st.columns([1, 2])
    with r2l:
        ph_fatigue = st.empty()
    with r2r:
        st.markdown(
            '<p class="feed-section-title">Mental Fatigue</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p class="bb-desc">A direct brainwave measure of tiredness — no AI involved. '
            'As you tire, slow theta waves rise and alert alpha waves fall.</p>',
            unsafe_allow_html=True)
        with st.popover("🔬 Technical Details"):
            st.markdown("""
**Reference:** Klimesch (1999) *Brain Research Reviews* 29:169-195

**Formula**
```
Fatigue = log(θ power) − log(α power)
θ = 4–8 Hz  (slow, drowsy)
α = 8–13 Hz (alert, relaxed)
```
Rising fatigue ≠ high workload — they can move independently.
Computed per-epoch from the raw EEG PSD, not from the model.
            """)
        ph_fat_pill = st.empty()

    st.markdown('<div class="bb-divider" style="margin:6px 0"></div>', unsafe_allow_html=True)

    # ── Row 3: Focus Streak ────────────────────────────────────────────────
    r3l, r3r = st.columns([1, 2])
    with r3l:
        ph_streak = st.empty()
    with r3r:
        st.markdown(
            '<p class="feed-section-title">Focus Streak</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p class="bb-desc">Tracks how long you\'ve stayed below the cognitive load '
            'threshold. A 3-window grace period keeps single noisy frames from '
            'breaking your streak.</p>',
            unsafe_allow_html=True)
        ph_streak_pill = st.empty()

    st.markdown('<div class="bb-divider" style="margin:6px 0"></div>', unsafe_allow_html=True)

    # ── Row 4: Signal Quality & Weights ───────────────────────────────────
    r4l, r4r = st.columns([1, 2])
    with r4l:
        ph_attn = st.empty()
    with r4r:
        st.markdown(
            '<p class="feed-section-title">Signal Quality & Weights</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p class="bb-desc">Shows which sensor the AI is relying on most. '
            'Spectral (frequency-band) features consistently generalize best '
            'across all subjects — watch this bar during a jaw clench.</p>',
            unsafe_allow_html=True)
        with st.popover("🔬 Technical Details"):
            st.markdown("""
**Signal quality index (Live mode)**
Viola et al. (2009) *J Neurosci Methods* 182(1):15-26
- std < 0.01 → flat line / disconnected electrode
- std > 3.0  → jaw clench / movement artefact

**Attention weights (all modes)**
Learned SQI gate: `att = softmax(content × (quality + 0.1) / 3)`

**The three towers:**
- **EEG** — raw brainwave amplitude from 4 electrodes (TP9, AF7, AF8, TP10)
- **Physio** — heart rate & skin conductance (Empatica E4; imputed in Live)
- **Spectral** — 36-dim log band power + β/(θ+α) + θ/α + frontal asymmetry

**Key finding:** σ(spectral accuracy) = 0.031 across 12 subjects —
lowest cross-subject variance of the three towers.
            """)
        ph_signal = st.empty()

    st.markdown('<div class="bb-divider" style="margin:6px 0"></div>', unsafe_allow_html=True)

    # ── Row 5: Trend (full width) ──────────────────────────────────────────
    st.markdown('<span class="bb-label">60-second workload &amp; fatigue trend</span>',
                unsafe_allow_html=True)
    ph_trend = st.empty()

    st.markdown('<div class="bb-divider" style="margin:6px 0"></div>', unsafe_allow_html=True)

    # ── Row 6: Classification accuracy ────────────────────────────────────
    ph_acc = st.empty()

    # ── Row 7: Raw sensor data (collapsed) ────────────────────────────────
    if not lite_mode:
        with st.expander("📡 Raw sensor data", expanded=False):
            wd1, wd2 = st.columns(2)
            with wd1:
                st.markdown('<span class="bb-label">EEG — 4 channels</span>',
                            unsafe_allow_html=True)
                ph_eeg = st.empty()
            with wd2:
                st.markdown('<span class="bb-label">Body signals — BVP / HR / EDA</span>',
                            unsafe_allow_html=True)
                ph_physio = st.empty()
    else:
        ph_eeg = ph_physio = None

    # ══════════════════════════════════════════════════════════════════════
    # SKELETON LOADERS — fill all placeholders now.
    # DO NOT REMOVE OR CONDITIONALISE THESE.
    # ══════════════════════════════════════════════════════════════════════
    ph_hero.markdown(
        f'<div style="min-height:110px">{_skel_bar(110)}</div>',
        unsafe_allow_html=True)
    ph_wl_pill.markdown(
        '<div class="skeleton-text" style="width:80px;height:24px;border-radius:100px"></div>',
        unsafe_allow_html=True)
    ph_fatigue.markdown(_skel_ring(), unsafe_allow_html=True)
    ph_fat_pill.markdown(
        '<div class="skeleton-text" style="width:80px;height:24px;border-radius:100px"></div>',
        unsafe_allow_html=True)
    ph_streak.markdown(
        f'<div style="text-align:center;padding:20px 10px">{_skel_bar(80)}</div>',
        unsafe_allow_html=True)
    ph_streak_pill.markdown(
        '<div class="skeleton-text" style="width:80px;height:24px;border-radius:100px"></div>',
        unsafe_allow_html=True)
    ph_attn.markdown(
        f'<div style="min-height:195px">{_skel_bar(195)}</div>',
        unsafe_allow_html=True)
    ph_signal.markdown(_skel_text_block(), unsafe_allow_html=True)
    ph_trend.markdown(
        f'<div style="min-height:200px">{_skel_bar(200)}</div>',
        unsafe_allow_html=True)
    ph_acc.markdown(
        '<div class="bb-card" style="text-align:center;padding:24px">'
        '<span class="bb-label">Press ▶ Start to begin</span></div>',
        unsafe_allow_html=True)
    if ph_eeg:
        ph_eeg.markdown(
            f'<div style="min-height:180px">{_skel_bar(180)}</div>',
            unsafe_allow_html=True)
    if ph_physio:
        ph_physio.markdown(
            f'<div style="min-height:180px">{_skel_bar(180)}</div>',
            unsafe_allow_html=True)

    # ── Accuracy badge helper ─────────────────────────────────────────────
    def _render_accuracy():
        if frag_demo and st.session_state.n_total > 0:
            acc     = st.session_state.n_correct / st.session_state.n_total * 100
            col     = "#00CC77" if acc>=60 else "#E09000" if acc>=45 else "#E05050"
            pcls    = "pill-green" if acc>=60 else "pill-orange" if acc>=45 else "pill-red"
            verdict = "ABOVE CHANCE" if acc>50 else "AT CHANCE" if acc>=45 else "BELOW CHANCE"
            ph_acc.markdown(f"""
            <div class="bb-card" style="display:flex;align-items:center;gap:24px;padding:16px 22px">
              <div>
                <span class="bb-label">CW Classification Accuracy</span>
                <span class="bb-value" style="color:{col};font-size:42px">{acc:.1f}%</span>
              </div>
              <div>
                <span class="bb-sub">{st.session_state.n_correct}/
                  {st.session_state.n_total} correct &nbsp;·&nbsp; chance = 50%</span><br>
                <span class="pill {pcls}" style="margin-top:8px;display:inline-block">
                  {verdict}</span>
              </div>
            </div>""", unsafe_allow_html=True)
        elif not frag_demo:
            ph_acc.markdown("""
            <div class="bb-card" style="padding:14px 22px">
              <span class="bb-label">Classification Accuracy</span>
              <span class="bb-value" style="color:#E09000;font-size:28px"> — </span>
              <span class="bb-sub">No ground truth in Live mode</span>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # TASK 1 FIX — Control buttons with corrected restart state machine.
    #
    # "↺ Restart Run" resets ALL state, then calls st.rerun() immediately.
    # This guarantees the fragment re-enters its loop with a clean slate —
    # no fall-through, no stale acquire_data flag, no dead loop.
    # ══════════════════════════════════════════════════════════════════════
    b1, b2, _ = st.columns([1, 1, 5])

    with b1:
        if st.session_state._run_complete:
            if st.button("↺ Restart Run", type="primary"):
                # Full state reset — mirrors _reset_history() plus frame/run flags
                st.session_state.frame_idx     = 0
                st.session_state._run_complete = False
                st.session_state._final_stats  = {}
                st.session_state.running       = True
                _reset_history()   # wipes wl_hist, fatigue_hist, counters, streak
                st.rerun()         # ← MUST be here; do not rely on fall-through
        else:
            if st.button("▶ Start",
                         disabled=st.session_state.running,
                         type="primary"):
                st.session_state.running = True
                st.rerun()

    with b2:
        if st.button("⏸ Pause" if st.session_state.running else "▶ Resume"):
            st.session_state.running = not st.session_state.running
            st.rerun()

    # ── Initialise render variables with frozen-frame fallbacks ──────────
    # These are overwritten below if acquisition runs this tick.
    smoothed_wl  = float(st.session_state.wl_hist[-1])
    smoothed_fat = float(st.session_state.fatigue_hist[-1])
    wl_color     = _wl_color(smoothed_wl)
    fat_color    = _fat_color(smoothed_fat)
    attn_vals    = np.array([0.33, 0.33, 0.34], dtype=np.float32)
    true_label   = 0
    pred         = 0
    p_threshold  = 0.5
    cw_prob      = 0.0
    eeg_ep       = np.zeros((4, 512), dtype=np.float32)
    physio_ep    = np.zeros((3, 512), dtype=np.float32)

    # ── Decide whether to run inference this tick ─────────────────────────
    should_run = (
        st.session_state.running
        and not st.session_state._run_complete
        and frag_model is not None
        and (not frag_demo or (f_labels is not None and f_eeg is not None))
    )

    if should_run:
        if frag_demo:
            i = st.session_state.frame_idx
            if i >= len(f_labels):
                # ── Dataset exhausted — commit stats and mark complete ────
                acc_final = (st.session_state.n_correct / st.session_state.n_total * 100
                             if st.session_state.n_total > 0 else 0.0)
                st.session_state._final_stats = {
                    "accuracy":      acc_final,
                    "n_correct":     st.session_state.n_correct,
                    "n_total":       st.session_state.n_total,
                    "mean_wl":       float(np.mean(list(st.session_state.wl_hist))),
                    "mean_fat":      float(np.mean(list(st.session_state.fatigue_hist))),
                    "focus_minutes": streak.minutes,
                }
                st.session_state._run_complete = True
                st.session_state.running       = False
                save_profile(st.session_state.user_id, st.session_state.profile)
                HISTORY.append({
                    "subject":       st.session_state.subject_idx,
                    "mean_wl":       st.session_state._final_stats["mean_wl"],
                    "mean_fat":      st.session_state._final_stats["mean_fat"],
                    "accuracy":      acc_final,
                    "focus_minutes": streak.minutes,
                })
                # Fall through to RENDER using frozen last-frame values
                should_run = False
            else:
                eeg_ep     = f_eeg[i]
                physio_ep  = f_physio[i]
                true_label = int(f_labels[i])
                freq_vec   = f_freq[i]
        else:
            # ── Live Muse path ────────────────────────────────────────────
            streamer = st.session_state.muse_streamer
            raw_ep   = streamer.get_epoch() if streamer else None
            if raw_ep is None:
                ph_hero.markdown(
                    '<div class="bb-card" style="text-align:center;padding:40px;min-height:110px">'
                    '<span class="pill pill-blue">⏳ Buffering Muse (~4 s)</span></div>',
                    unsafe_allow_html=True)
                time.sleep(0.5); st.rerun(); return

            eeg_ep = np.clip(
                (raw_ep - raw_ep.mean(axis=1, keepdims=True))
                / (raw_ep.std(axis=1, keepdims=True) + 1e-8), -5, 5)

            stable, reason = is_signal_stable(eeg_ep)
            st.session_state.signal_stable = stable
            st.session_state.signal_reason = reason
            st.session_state.stable_count  = (
                min(st.session_state.stable_count + 1, 99) if stable else 0)
            if st.session_state.stable_count < 4:
                ph_signal.markdown(
                    f'<span class="pill pill-orange">⚠ {reason}</span>',
                    unsafe_allow_html=True)
                time.sleep(0.5); st.rerun(); return

            physio_ep  = np.zeros((3, 512), dtype=np.float32)
            true_label = -1
            freqs, psd = welch(eeg_ep, fs=256.0, nperseg=256, axis=1)
            bands = [(1,4),(4,8),(8,13),(13,30),(30,40)]
            bp    = [np.clip(np.mean(psd[:,(freqs>=lo)&(freqs<hi)],axis=1),1e-12,1e3)
                     for lo,hi in bands]
            pmat     = np.stack(bp, axis=1)
            freq_vec = np.concatenate([
                np.log(pmat).flatten(),
                np.log(pmat[:,3]/(pmat[:,1]+pmat[:,2]+1e-12)),
                np.log(pmat[:,1]/(pmat[:,2]+1e-12)),
                np.log(pmat[2,:]+1e-12)-np.log(pmat[1,:]+1e-12),
                np.zeros(3, dtype=np.float32)])

    if should_run:
        # ── Mental Fatigue (Klimesch 1999) — spectral, no model ──────────
        theta_log      = float(np.mean(freq_vec[4:8]))
        alpha_log      = float(np.mean(freq_vec[8:12]))
        fatigue_scaled = float(np.clip((theta_log - alpha_log + 1.5) * 40, 0, 100))

        # ── CW Inference ──────────────────────────────────────────────────
        with torch.inference_mode():
            logits, attn_t = frag_model(
                torch.from_numpy(eeg_ep[None,None,:,:]).float().to(DEVICE),
                torch.from_numpy(physio_ep[None,:,:]).float().to(DEVICE),
                torch.from_numpy(freq_vec[None,:]).float().to(DEVICE))
            cw_prob = torch.softmax(logits, dim=1)[0, 1].item()

        attn_vals  = attn_t[0].cpu().numpy()
        attn_vals /= np.sum(attn_vals) + 1e-8

        # ── Write debug to session_state (sidebar reads outside fragment) ─
        st.session_state._dbg_cw        = cw_prob
        st.session_state._dbg_frame     = st.session_state.frame_idx
        st.session_state._dbg_freq_mean = float(freq_vec.mean())
        st.session_state._dbg_freq_std  = float(freq_vec.std())

        # ── Personalization (Live only) ────────────────────────────────────
        if not frag_demo:
            st.session_state.profile  = update_profile(st.session_state.profile, cw_prob)
            st.session_state.fss     += 1
            if st.session_state.fss >= 10:
                save_profile(st.session_state.user_id, st.session_state.profile)
                st.session_state.fss = 0

        profile = st.session_state.profile

        # ── Workload + threshold ───────────────────────────────────────────
        # Demo: Workload = 100 × P(high CW)  (0% rested, 100% maxed out)
        # Live: personalised normalisation over user's personal CW range
        if frag_demo:
            workload    = cw_prob * 100.0
            p_threshold = 0.5
        else:
            workload    = personalized_workload(cw_prob, profile)
            p_threshold = personalized_threshold(profile)

        pred = 1 if cw_prob >= p_threshold else 0
        if frag_demo:
            st.session_state.n_total   += 1
            st.session_state.n_correct += int(pred == true_label)

        # ── Focus streak update ────────────────────────────────────────────
        streak.update(cw_prob < p_threshold)
        st.session_state.streak = streak

        # ── EMA smoothing ──────────────────────────────────────────────────
        prev_wl      = st.session_state.wl_hist[-1]
        smoothed_wl  = float(np.clip(prev_wl  + ema_alpha*(workload       - prev_wl),  0, 100))
        prev_fat     = st.session_state.fatigue_hist[-1]
        smoothed_fat = float(np.clip(prev_fat + ema_alpha*(fatigue_scaled  - prev_fat), 0, 100))

        frame_t = st.session_state.frame_idx if frag_demo else len(st.session_state.wl_hist)
        st.session_state.wl_hist.append(smoothed_wl)
        st.session_state.fatigue_hist.append(smoothed_fat)
        st.session_state.time_hist.append(frame_t)

        wl_color  = _wl_color(smoothed_wl)
        fat_color = _fat_color(smoothed_fat)

        if frag_demo:
            st.session_state.frame_idx += 1

    # Re-read streak (needed for frozen render path after run-complete)
    streak = st.session_state.get("streak", FocusStreak())

    # ════════════════════════════════════════════════════════════════════
    # RENDER — always executes.
    # Uses live values when should_run=True, frozen values otherwise.
    # ════════════════════════════════════════════════════════════════════

    # Row 1: Workload hero bar
    gt_str = ""
    if frag_demo and show_gt and not st.session_state._run_complete:
        gt_str = "GT: Hi-load" if true_label else "GT: Calm"
    ph_hero.markdown(make_workload_bar_html(smoothed_wl, gt_str), unsafe_allow_html=True)
    ph_wl_pill.markdown(_wl_pill(smoothed_wl), unsafe_allow_html=True)

    # Row 2: Fatigue ring
    fat_lbl = "ALERT" if smoothed_fat<35 else "TIRED" if smoothed_fat<65 else "FATIGUED"
    ph_fatigue.plotly_chart(
        make_ring(smoothed_fat, fat_color, fat_lbl, "θ/α ratio"),
        use_container_width=True, key="fatigue_ring")
    fat_pill_cls = "pill-green" if smoothed_fat<35 else "pill-orange" if smoothed_fat<65 else "pill-red"
    ph_fat_pill.markdown(
        f'<span class="pill {fat_pill_cls}">{fat_lbl}</span>'
        f'<span class="bb-sub" style="margin-left:8px">{smoothed_fat:.0f}/100</span>',
        unsafe_allow_html=True)

    # Row 3: Focus streak
    streak_col    = "#00CC77" if streak.seconds > 0 else "#4B5563"
    streak_status = "FOCUSING" if streak.seconds > 0 else "RESET"
    streak_pcls   = "pill-green" if streak.seconds > 0 else "pill-dim"
    ph_streak.markdown(f"""
    <div style="text-align:center;padding:18px 10px;min-height:110px">
      <div class="streak-label">Focus Streak</div>
      <div class="streak-display" style="color:{streak_col};margin:8px 0">
        {streak.display}</div>
      <div class="streak-label">MM:SS</div>
    </div>""", unsafe_allow_html=True)
    ph_streak_pill.markdown(
        f'<span class="pill {streak_pcls}">{streak_status}</span>'
        f'<span class="bb-sub" style="margin-left:8px">{streak.minutes:.1f} min this session</span>',
        unsafe_allow_html=True)

    # Row 4: Attention bar + signal quality
    ph_attn.plotly_chart(
        make_attention_chart(attn_vals),
        use_container_width=True, key="attn")

    if frag_demo:
        sq_pct = min(float(attn_vals[2]) * 300, 100)
        sq_col = "#00CC77" if sq_pct>60 else "#E09000" if sq_pct>30 else "#E05050"
        sq_cls = "pill-green" if sq_pct>60 else "pill-orange" if sq_pct>30 else "pill-red"
        sq_lbl = "GOOD" if sq_pct>60 else "MODERATE" if sq_pct>30 else "POOR"
        ph_signal.markdown(
            f'<span class="pill {sq_cls}">SQI {sq_lbl}</span>'
            f'<span class="bb-sub" style="margin-left:8px">'
            f'Spectral: {attn_vals[2]:.2f} &nbsp;|&nbsp; '
            f'EEG: {attn_vals[0]:.2f} &nbsp;|&nbsp; '
            f'Physio: {attn_vals[1]:.2f}</span>',
            unsafe_allow_html=True)
    else:
        stable = st.session_state.get("signal_stable", True)
        reason = st.session_state.get("signal_reason", "OK")
        sq_cls = "pill-green" if stable else "pill-red"
        sq_lbl = "STABLE" if stable else "UNSTABLE"
        ph_signal.markdown(
            f'<span class="pill {sq_cls}">{sq_lbl}</span>'
            f'<span class="bb-sub" style="margin-left:8px">{reason}</span>',
            unsafe_allow_html=True)

    # Row 5: Trend chart
    ph_trend.plotly_chart(
        make_trend_chart(list(st.session_state.time_hist),
                         list(st.session_state.wl_hist),
                         list(st.session_state.fatigue_hist), p_threshold),
        use_container_width=True, key="trend")

    # Row 6: Accuracy
    _render_accuracy()

    # Row 7: Raw waveforms
    if ph_eeg is not None:
        ph_eeg.plotly_chart(make_eeg_chart(eeg_ep),
                             use_container_width=True, key="eeg")
    if ph_physio is not None:
        if frag_demo:
            ph_physio.plotly_chart(make_physio_chart(physio_ep),
                                    use_container_width=True, key="physio")
        else:
            ph_physio.markdown(
                '<div style="text-align:center;padding:40px;color:#4B5563;font-size:13px">'
                'No Empatica E4 — EEG-only inference active</div>',
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # LOOP DRIVER
    # ─────────────────────────────────────────────────────────────────────
    # Call st.rerun() whenever running=True, even if no inference happened
    # this tick (e.g. first tick after a subject switch or restart where
    # should_run was temporarily False). This prevents the fragment from
    # going idle when it should be looping.
    # ══════════════════════════════════════════════════════════════════════
    if st.session_state.running:
        time.sleep(speed)
        st.rerun()


live_display()
