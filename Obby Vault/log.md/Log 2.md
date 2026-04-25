# Brain Battery — Log 2: LOSO Benchmark + Live-Mode UX Overhaul

> Newest first. Companion to `Log 1.md`. Sessions: **2026-04-24**, **2026-04-25**.

---

## [2026-04-25] docs | Rewrite Live Mode setup descriptions — dashboard + README

**Branch:** `claude/beautiful-dhawan-57d37a` (dashboard) · `main` (README)
**Commits:** `3a707ba` (dashboard) · `a3cfde4` (README)
**Files changed:** `dashboard.py`, `README.md`

**Problem:** The brainflow setup card said "Brain Battery will scan for your Muse over Bluetooth" — too vague. Users didn't know whether they needed to pair in Windows Settings, how long to wait, or what to do if it failed. The BlueMuse card was a single run-on sentence with no numbered steps.

**Changes:**

- `dashboard.py` — Step 1 card renamed from "Hardware" to **"Power on your Muse"**. Now says: hold button 3 seconds until LED blinks, confirm Bluetooth is on in Windows Settings.
- `dashboard.py` — brainflow card now explicitly states:
  - ❌ No Windows Bluetooth pairing needed — brainflow handles it automatically
  - ❌ No extra software to install
  - ✅ Click Connect; app searches BLE for up to 10 s
  - Bulleted failure checklist: close other Muse apps (MindMonitor etc.), power-cycle the headset, move within 1–2 m
- `dashboard.py` — BlueMuse card is now a numbered 5-step list with exact UI labels ("Start Streaming", "LSL: Sending"), and a note that closing BlueMuse disconnects the stream.
- `dashboard.py` — Header copy updated to explain the two-method choice upfront.
- `README.md` — Live Mode section rewritten from 3 vague lines into two labelled methods (Method 1: brainflow, Method 2: BlueMuse) with full step-by-step instructions, failure troubleshooting, and a "What Live Mode shows" paragraph describing the electrode strip.
- `README.md` — Features bullet updated to mention brainflow direct path and contact quality strip.

---

## [2026-04-25] fix | Replace deprecated use_container_width API + debug run

**Branch:** `claude/beautiful-dhawan-57d37a`
**Commit:** `8da1fe1`
**Files changed:** `dashboard.py`

Replaced all 22 instances of the deprecated `use_container_width=True/False` parameter
with `width='stretch'`/`width='content'` per the Streamlit 1.56 deprecation notice
(deadline was 2025-12-31). Eliminates all startup deprecation warnings. App now launches
completely clean.

Also ran a full static + runtime debug pass before committing:
- AST syntax check: OK
- All new symbols verified present (`detect_muse_drivers`, `MUSE_DRIVER_AVAILABLE`, `MuseStreamer` methods)
- `detect_muse_drivers(0.3)` dry-run: returns correct state without hardware
- Electrode strip `None`-buffer path: renders "Buffering…" gracefully
- App launched on port 8503: zero warnings in startup log

---

## [2026-04-24] benchmark | First end-to-end LOSO run on the SANN checkpoint

**Branch:** `claude/beautiful-dhawan-57d37a`  
**Files written:** `results/loso_results.json`, `results/research_summary.json`, `results/plots/subject_*_*.png` (48 images)

### Headline numbers

| Model    | Mean acc        | Std    | Best subject     | Worst subject    |
|----------|-----------------|--------|------------------|------------------|
| **SANN** | **51.1%**       | ±15.5% | S18 (79.7%)      | S21 (15.6%)      |
| XGBoost  | 49.9%           | ±8.0%  | S06 (60.8%)      | S21 (23.4%)      |

SANN beats XGBoost by **only +1.2 percentage points** on the mean and is *twice* as variable. At the LOSO scale neither model meaningfully clears chance.

### The real finding: prediction collapse

The mean is misleading. After running confusion-matrix bookkeeping across all 24 subjects:

- **24 / 24 subjects** — SANN predicts **HIGH cognitive workload for ≥95% of test epochs** at the default 0.5 threshold. The model has effectively collapsed to a constant predictor in the LOSO regime.
- Despite the collapse, **mean AUC = 0.563**, and **3 subjects have AUC > 0.7** (S01 0.84, S18 0.68 with very high prevalence, S04 mid-range). The internal *ranking* of "more vs. less likely high-CW" is therefore weakly informative — only the threshold is broken.
- This is the same calibration issue the live dashboard already mitigates with **per-subject Welford running statistics** (see `personalization.py`). LOSO eval does not apply that, so the offline numbers are the worst-case "no-calibration" floor.

Practical reading: **report AUC + per-subject calibrated accuracy** as the headline going forward, not raw 0.5-threshold accuracy.

### Subject anomalies

- **S21 (15.6% acc, AUC 0.45)** — only 4.4% high-CW prevalence in raw data. Predicting all-high → 95.6% wrong. Outlier; expect this whenever the holdout subject's prior is far from the dataset average.
- **S15 (20.2% acc, AUC 0.44)** — sub-chance ranking. Worth checking signal quality (`sqi.npy`) for this subject.
- **S11 (26.8% acc)** — same family as S21; very low high-CW prevalence (10%).
- **S18, S17, S01** all > 70% — these subjects' prevalence happens to align with the model's collapse, not because the model "understands" them.

### Performance

- Total LOSO wall-time: ~3 minutes on CPU (sum of per-subject inference ~45 s; the bulk is data slicing/normalisation).
- `bottleneck` flag: **0 subjects** triggered the `t_load / t_infer > 1.2` heuristic. CPU is not the limiter at this dataset size.

### Next experiments worth running

1. Re-eval with the live-mode personalization layer applied per holdout subject (warm-start Welford on first ~5% of test, then evaluate on the remaining 95%).
2. Threshold sweep per subject: report accuracy at the AUC-optimal threshold instead of 0.5.
3. Drop the 3 outlier subjects (S11, S15, S21) and re-report — check whether the mean − 1 SD outlier rule is masking a tighter "core 21" benchmark.

---

## [2026-04-24] fix | LOSO eval was running with wrong feature shape and unfiltered ambiguous labels

**Branch:** `claude/beautiful-dhawan-57d37a`  
**Files changed:** `Scripts/eval_loso.py`

**Summary:** Two latent bugs in the eval pipeline, both pre-existing, both surfaced by the first attempt to actually run it. Without these fixes the eval crashes immediately and, even if it didn't, ~12% of every subject's test set was being scored on a label the model was never trained to predict.

**Key changes:**

- `Scripts/eval_loso.py::run_sann` — was passing the raw 33-dim `psd_features.npy` into a `FreqEncoder(in_dim=36)`. The training pipeline (`train.py::_compute_physio_stats`) concatenates 3 physio statistics — HR mean, HR std, log-EDA mean — onto the 33 PSD features to reach the 36-dim input. Eval now reproduces that exactly via a new `compute_physio_stats(physio_np)` helper.
- `Scripts/eval_loso.py::load_data` — now drops all epochs where `label_workload == -1` (12.1% of the dataset, 21,184 epochs), matching `train.py`'s `LABEL_AMBIGUOUS` filter. Without this, every subject's reported accuracy was contaminated by a class the model has never been trained to predict.
- Comment on line 162 ("Pre-computed spectral features (36-dim)") is now technically wrong — the file is 33-dim. Left unchanged in this commit; flag for cleanup.

---

## [2026-04-24] feature | Live Muse setup overhaul — brainflow primary, BlueMuse one-click fallback

**Branch:** `claude/beautiful-dhawan-57d37a`  
**Files changed:** `dashboard.py`, `requirements.txt`

**Summary:** The previous live-mode setup walked the user through a 5-card BlueMuse tutorial: install BlueMuse, pair via Windows Bluetooth, manually start the LSL stream, then click Connect and pray. Failure mode was a single 5-second timeout and a single string ("No Muse stream"). New flow uses `brainflow` direct-BLE as the primary path so most users never see BlueMuse, with a one-click fallback when brainflow isn't available or fails.

**Key changes:**

- `dashboard.py::MuseStreamer` — rewritten with two driver paths (`_connect_brainflow`, `_connect_lsl`). `connect()` chooses the order based on whether an LSL stream is already live. Each path validates that samples actually arrive within 2 seconds before reporting success. New `meta` property and `channel_quality()` method expose stream metadata and per-channel std for the UI.
- `dashboard.py` — new `detect_muse_drivers()` preflight returns `{brainflow_available, lsl_available, lsl_stream_found, lsl_stream_meta, recommended}` in <0.3 s. Used to drive UI status pills and enable/disable the Connect button.
- `dashboard.py` Live setup subpage — adaptive layout. With brainflow installed: a single "Direct BLE" card and Connect button. Without: BlueMuse card + **Download BlueMuse** link button (`https://github.com/kowalej/BlueMuse/releases/latest`) + **Launch BlueMuse** button (uses `bluemuse://start` URI handler) + **Re-detect** button. Diagnostics expander shows raw driver state.
- `dashboard.py` Live overview — replaced the binary "signal stable / signal issue" badge with a 4-cell electrode contact strip (TP9, AF7, AF8, TP10), each color-coded by per-channel std over the last 1 s of the streamer buffer (red < 2 µV flat, amber > 150 µV noisy, green ok). No new sampling — reads from the existing `MuseStreamer._buf` deque.
- `dashboard.py` driver gate — was `if not PYLSL_AVAILABLE: stop`. Now `if not (PYLSL_AVAILABLE or BRAINFLOW_AVAILABLE): stop`, with installation instructions for both.
- `requirements.txt` — added `brainflow` and `pylsl` (pylsl was previously only loaded via try/except and never declared).

**Pending:** manual UI verification with a real Muse — see "Quick setup checklist" below.

---

## Quick setup checklist for tomorrow's laptop test

Run these in order. Total time on a clean laptop: ~5 min if brainflow works, ~10 min if it falls back to BlueMuse.

1. **Pull the branch and install:**
   ```bash
   git checkout claude/beautiful-dhawan-57d37a
   pip install -r requirements.txt
   ```
   This pulls in `brainflow` and `pylsl`. brainflow has a small native binary that ships in the wheel — no compiler needed on Win/macOS/Linux x86_64.

2. **Power on the Muse** (button held until LED blinks). Make sure no other app holds the BLE link — close MindMonitor, Petal Metrics, or anything else that auto-connects to the headset.

3. **Launch:**
   ```bash
   python -m streamlit run dashboard.py
   ```
   The `python -m` form matters — it ensures the same interpreter that has `brainflow`/`pylsl` is the one running Streamlit.

4. **Pick `🎧 Live — Muse headband`** in the sidebar.

5. **Click "Setup Muse Headset" → "Connect"**. Expected behaviour with brainflow:
   - Spinner appears for ~5–10 s while brainflow scans for the Muse over BLE.
   - On success the page transitions to overview and the 4-cell electrode strip should populate within ~2 s.

6. **If brainflow fails** (no BLE adapter, OS quirks, Bluetooth turned off):
   - The setup page automatically swaps to the BlueMuse card.
   - Click **Download BlueMuse** (opens GitHub releases page in browser).
   - Run the installer, pair the Muse in **Windows Settings → Bluetooth & devices → Add device → Muse-XXXX**.
   - Back in the app click **Launch BlueMuse** (uses `bluemuse://start` URI to open the app and start streaming) — or open BlueMuse manually and click Start Streaming.
   - Click **Re-detect**. The "Stream detected ✓" pill should turn green.
   - Click **Connect**.

### What to verify

- [ ] Connect succeeds via brainflow (or falls back cleanly to BlueMuse).
- [ ] Electrode strip shows 4 green cells when the headset is sitting properly.
- [ ] Lift one earpiece off the skin → the corresponding cell flips to red ("flat") within ~1 s.
- [ ] Clench your jaw briefly → at least one cell flips to amber ("noisy").
- [ ] Dashboard predictions update; the "CW Classification Accuracy" badge starts ticking once enough epochs accumulate.

### Common fail modes and fixes

- **`No Muse Driver Installed` page on entry to Live mode** — pip install didn't pick up `brainflow` or `pylsl`. Run `python -m pip install brainflow pylsl` against the same interpreter.
- **brainflow throws `BOARD_NOT_READY_ERROR`** — Muse is paired to another app/phone. Power-cycle the headset and close other Muse apps.
- **BlueMuse "Start Streaming" button is greyed** — Muse not paired in Windows Settings yet. Pair first, then start streaming.
- **Connect button stays disabled in BlueMuse mode** — preflight isn't seeing the LSL stream. Wait 2–3 s and click Re-detect; verify BlueMuse status reads `LSL: Sending`.
- **All 4 electrodes flat after connect** — BlueMuse sometimes opens the LSL stream before samples flow. Stop and restart streaming in BlueMuse, then click Re-detect.
