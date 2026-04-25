> LLM Wiki — Brain Battery Dashboard Last updated: 2026-04-16

---

## Cognitive Workload (Hero Metric)

**What it measures:** Mental effort required by a task. **What it is NOT:** Emotional stress (different labels, different biomarkers).

```python
# Raw model output
P(high CW) = softmax( workload_head( EEG ⊕ Physio ⊕ Freq ) )₁

# Hero metric (v4.2+)  ← reversed from v4.1 bandwidth
Workload = 100 × P(high CW)
# 0%  = fully rested
# 100% = brain maxed out
```

**Colour scale (v4.2+):**

|Range|Color|Label|Hex|
|---|---|---|---|
|0–30%|Green|LOW EFFORT|`#00CC77`|
|30–70%|Amber|ACTIVE|`#E09000`|
|70–100%|Red|HEAVY LOAD|`#E05050`|

**Threshold (Demo mode):** Fixed at 0.5 → Workload threshold = 50%. **Threshold (Live mode):** Personalized via Welford algorithm (see below).

**Previous version note:** v4.1 used `Bandwidth = 100×(1−P)` where high = good. v4.2 reversed this to `Workload = 100×P` where high = bad. The variable `bw_hist` was globally renamed to `wl_hist` and `mean_bw` → `mean_wl`.

Source: [[raw/Lawhern_2018_EEGNet.pdf]] · [[raw/Holm_2009_beta_ratio.pdf]]

---

## Mental Fatigue (Spectral — No Model)

```python
# Klimesch (1999) theta/alpha ratio
Fatigue = log(θ_power) − log(α_power)

# Band definitions:
θ = 4–8 Hz   (drowsy, slow)
α = 8–13 Hz  (alert, relaxed)

# Scaled to 0–100 for display:
fatigue_scaled = clip((theta_log - alpha_log + 1.5) × 40, 0, 100)
```

Rising fatigue ≠ high workload. They can move independently. Computed from raw EEG PSD per epoch — the model is not involved.

Source: [[raw/Klimesch_1999.pdf]]

---

## EMA Smoothing

```python
# Applied to both workload and fatigue every inference tick
display[t] = α × model_output[t] + (1 − α) × display[t−1]

# α = 0.7 (default): reacts within ~3 frames (~6 seconds)
# α = 1.0: instant, noisy
# α = 0.1: very smooth, slow to react
```

Source: Standard exponential moving average (no citation required)

---

## Personalized Workload (Live Mode)

```python
def personalized_workload(cw_prob: float, profile: dict) -> float:
    """
    Map raw P(high CW) to 0–100 using user's personal CW range.
    High output = high workload = bad.
    """
    lo, hi = profile["range_min"], profile["range_max"]
    return round(clip((cw_prob - lo) / max(hi - lo, 0.05), 0, 1) * 100.0, 1)
```

**Threshold:**

```python
def personalized_threshold(profile: dict) -> float:
    """threshold = μ + 0.5σ, clamped [0.30, 0.80]"""
    return clip(profile["mean"] + 0.5 × personalized_std(profile), 0.30, 0.80)
```

Source: [[raw/Welford_1962.pdf]]

---

## Welford Online Mean/Variance

```python
# Numerically stable running statistics — no catastrophic cancellation
def update_profile(profile: dict, cw_prob: float) -> dict:
    profile["count"] += 1
    delta = cw_prob - profile["mean"]
    profile["mean"] += delta / profile["count"]
    profile["M2"]   += delta * (cw_prob - profile["mean"])
    # ...range tracking with slow EMA (α=0.02)
    return profile
```

Source: [[raw/Welford_1962.pdf]]

---

## Signal Quality Index (SQI)

```python
# Live mode — per-channel std heuristic after instance normalisation
def is_signal_stable(eeg_epoch: np.ndarray) -> Tuple[bool, str]:
    ch_std = eeg_epoch.std(axis=1)
    if any(ch_std < 0.01): return False, "Flat signal"    # disconnected
    if any(ch_std > 3.0):  return False, "Movement artefact"
    return True, "OK"
```

**Demo mode proxy:** `SQI_proxy = min(attn_vals[spectral] × 300, 100)` Spectral attention weight ∝ signal quality — when artefacts pollute EEG, the model down-weights the EEG tower and up-weights spectral features.

Source: [[raw/Viola_2009_SQI.pdf]]

---

## Attention Fusion (SQI Gate)

```python
# att = softmax(content_logits × (quality + 0.1) / 3.0)
# Three towers: EEG, Physio, Spectral
# quality ∈ (0,1)^3 from learned SQI projection
```

**Key finding (UNIVERSE, n=12, LOSO):** σ(spectral accuracy) = 0.031 — lowest cross-subject variance of the three towers. Spectral features generalize best across individuals.

Source: [[raw/Lawhern_2018_EEGNet.pdf]] · [[raw/Schirrmeister_2017.pdf]]

---

## Model Architecture Summary

```
Input EEG (B,1,4,512) ─▶ EEGNet ──────────────────────▶ f_eeg  (B,32)
Input Physio (B,3,512) ─▶ PhysioCNN ──────────────────▶ f_phys (B,32)
Input Freq (B,36)      ─▶ FreqEncoder ────────────────▶ f_freq (B,32)
SQI (B,6)              ─▶ SQIGate ────▶ quality (B,3)
                                              │
f_eeg, f_phys, f_freq ──▶ AttentionFusion ◀──┘
                                 │
                         fused (B,96) ──▶ workload_head ──▶ P(low, high) (B,2)
                                                                    │
                                                           softmax[:,1] = P(high CW)
                                                                    │
                                                   Workload = 100 × P(high CW)
```

TOWER_DIM = 32, FUSION_DIM = 96 (3 × TOWER_DIM)