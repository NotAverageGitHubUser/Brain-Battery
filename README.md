<div align="center">

# Brain Battery

### Real-time Cognitive Workload Monitor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/brandon19834/universe-merged-withzero-noasr)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*Measures cognitive workload, mental fatigue, and focus streaks in real time using EEG — powered by a custom multimodal deep learning model trained on the UNIVERSE dataset (n=24 subjects, LOSO).*

**[Website](https://notaveragegithubuser.github.io/Brain-Battery) · [Dataset](https://www.kaggle.com/datasets/brandon19834/universe-merged-withzero-noasr) · [Getting Started](#getting-started)**

<!-- Add your screenshot here: ![Brain Battery Dashboard](docs/assets/screenshot-demo.png) -->

</div>

---

## What Brain Battery Measures

| Metric | How | Notes |
|--------|-----|-------|
| **Cognitive Workload** | Multimodal neural network (EEG + Physio + Spectral features) | NASA-TLX binary labels · UNIVERSE dataset |
| **Mental Fatigue** | log θ − log α power ratio | Klimesch (1999) · EEG spectral only |
| **Focus Streak** | Continuous low-CW windows · 3-window grace period | Resets only after sustained high load |
| **Signal Quality** | Per-channel σ heuristic (live) · Spectral attention proxy (demo) | Viola et al. (2009) |

> **Important:** Brain Battery measures *cognitive workload* — sustained mental demand from tasks. It does **not** measure emotional stress (different construct, different labels, different biomarkers).

---

## Features

- **Demo mode** — replay 24 pre-recorded subjects from the UNIVERSE EEG + physio dataset with full playback controls
- **Live mode** — connect a Muse 2 or Muse S headband directly over Bluetooth (brainflow, no extra app) or via [BlueMuse](https://github.com/kowalej/BlueMuse) as a fallback; real-time EEG inference with per-electrode contact quality strip
- **Personalization** — online Bayesian calibration adapts to your individual cognitive baseline over sessions
- **Session history** — every run is logged; trend charts and per-session summaries stored locally
- **First-run setup screen** — auto-downloads the dataset from Google Drive (no account needed) or Kaggle

---

## Architecture

The model is a three-tower multimodal network trained with Leave-One-Subject-Out (LOSO) cross-validation:

```
EEG (4-ch × 256 samples)   →  EEGNet (depthwise/separable convolutions)  ─┐
Physio (BVP + HR + EDA)     →  1D-CNN                                      ├→ Attention Fusion → Workload Head → P(high CW)
Spectral (36 features)      →  MLP with BatchNorm                          ─┘
```

- **EEGNet** — Lawhern et al. (2018): compact depthwise-separable CNN purpose-built for EEG
- **Attention fusion** — learned per-modality weights conditioned on signal quality
- **Workload head** — two-layer MLP with dropout; binary softmax output
- **LOSO accuracy** — generalizes across subjects without per-subject calibration

---

## Getting Started

### Prerequisites

- Python 3.10+
- All other dependencies install automatically via `requirements.txt`

### Quickstart

```bash
git clone https://github.com/NotAverageGitHubUser/Brain-Battery.git
cd Brain-Battery
pip install -r requirements.txt
python -m streamlit run dashboard.py
```

On first run, a setup screen appears automatically. Click **"Download from Google Drive"** — no account or API key required. The app downloads the dataset (~300 MB) and launches.

### Dataset

The UNIVERSE EEG dataset is hosted publicly on Google Drive:

**[📁 Download data folder →](https://drive.google.com/drive/folders/1EiRr-hOapvOXsaHG16uW5Be1FZ3146Q6?usp=drive_link)**

The in-app setup screen handles this automatically, but if you prefer to download manually, copy these four files into a `Data/` folder next to `dashboard.py`:

`eeg.npy` · `physio.npy` · `label_workload.npy` · `subjects.npy`

Also available on Kaggle: [`brandon19834/universe-merged-withzero-noasr`](https://www.kaggle.com/datasets/brandon19834/universe-merged-withzero-noasr)

### Run

```bash
python -m streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser. Choose **Demo Mode** to explore pre-recorded subjects, or **Live Mode** if you have a Muse headband.

---

## Dataset

The UNIVERSE dataset contains EEG and physiological recordings from 24 subjects performing cognitive tasks at varying mental load levels, labeled with NASA Task Load Index (NASA-TLX) scores.

| Property | Value |
|----------|-------|
| Subjects | 24 |
| EEG channels | 4 (TP9, AF7, AF8, TP10) |
| Physio signals | BVP, HR, EDA |
| Labels | Binary workload (NASA-TLX) |
| Epoch length | 256 samples |
| Validation | Leave-One-Subject-Out (LOSO) |
| Published | *Scientific Data* (Nature, 2024) |

[**Download on Kaggle →**](https://www.kaggle.com/datasets/brandon19834/universe-merged-withzero-noasr) &nbsp;·&nbsp; [**Original dataset repo →**](https://github.com/HPI-CH/UNIVERSE) &nbsp;·&nbsp; [**Nature paper →**](https://www.nature.com/articles/s41597-024-03738-7)

---

## Live Mode (Muse Headband)

Live mode streams EEG from a [Muse 2 or Muse S](https://choosemuse.com/) headband directly into the model. Two connection methods are supported — try **Direct (brainflow)** first, fall back to **BlueMuse** if needed.

> **Note:** Live mode uses EEG only (TP9, AF7, AF8, TP10). Physiological signals (BVP, HR, EDA) are not captured by the Muse hardware — they are available in Demo mode from the pre-recorded dataset.

### Method 1 — Direct Bluetooth via brainflow (recommended, no extra app)

`brainflow` is already included in `requirements.txt`, so if you ran `pip install -r requirements.txt` you already have it.

1. **Turn on your Muse** — hold the button on the back for 3 seconds until the LED blinks white.
2. **Make sure Bluetooth is on** on your laptop (Windows Settings → Bluetooth & devices → toggle ON).
3. **Close any other app** that might be connected to the Muse right now (MindMonitor, Petal Metrics, the official Muse app). These hold an exclusive Bluetooth lock and will block the connection.
4. **Launch Brain Battery** and select **🎧 Live — Muse headband** in the sidebar.
5. Click **Setup Muse Headset** → **Connect**. Brain Battery searches for your Muse over Bluetooth for up to 10 seconds and connects automatically. No pairing in Windows Settings required.

**If Connect fails:** power-cycle the Muse (hold button until LED off, wait 3 s, hold again until it blinks), move within 1–2 m of your laptop, and try again.

### Method 2 — BlueMuse LSL bridge (fallback, Windows only)

Use this if the direct method doesn't work on your machine (some Windows configurations block direct BLE access from Python).

1. **Download and install [BlueMuse](https://github.com/kowalej/BlueMuse/releases/latest)** — free Windows app (~50 MB installer).
2. **Pair the Muse in Windows Settings** — Settings → Bluetooth & devices → Add device → select **Muse-XXXX** from the list (one-time setup, ~10 seconds).
3. **Open BlueMuse** and click **Start Streaming**. Wait for the status to read **LSL: Sending** (green text). Keep BlueMuse running in the background.
4. **Launch Brain Battery** and select **🎧 Live — Muse headband** → **Setup Muse Headset**. The setup page detects the stream automatically and enables the Connect button.
5. Click **Connect**.

### What Live Mode shows

Once connected, the dashboard shows a **4-electrode contact strip** (TP9 / AF7 / AF8 / TP10) with live signal quality for each electrode — green means good contact, red means the electrode is flat (not touching skin), amber means movement noise. The model runs inference on every 2-second EEG window and updates the Cognitive Workload percentage in real time.

---

## References

```
Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance.
  Brain Research Reviews, 29, 169–195.

Lawhern, V. J. et al. (2018). EEGNet: A compact convolutional neural network for EEG-based
  brain-computer interfaces. J. Neural Engineering, 15(5).

Holm, A. et al. (2009). Psychophysiological performance metrics for adaptive automation
  during simulated flight. Psychophysiology, 46(5), 583–598.

Schirrmeister, R. T. et al. (2017). Deep learning with convolutional neural networks for
  EEG decoding and visualization. Human Brain Mapping, 38(9), 4456–4473.

Viola, F. C. et al. (2009). Semi-automatic identification of independent components
  representing EEG artifact. J. Neuroscience Methods, 182(1), 15–26.

Welford, A. T. (1962). On changes of performance with age.
  Technometrics, 4(3), 419–420.
```

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
<sub>Built by Brandon Dong · brandondong999@gmail.com</sub>
</div>
