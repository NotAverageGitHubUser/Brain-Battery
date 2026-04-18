<div align="center">

# Brain Battery

### Real-time Cognitive Workload Monitor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/brandon19834/universe-merged-withzero-noasr)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*Measures cognitive workload, mental fatigue, and focus streaks in real time using EEG — powered by a custom multimodal deep learning model trained on the UNIVERSE dataset (n=12 subjects, LOSO).*

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

- **Demo mode** — replay 12 pre-recorded subjects from the UNIVERSE EEG + physio dataset with full playback controls
- **Live mode** — connect a Muse headband via [BlueMuse](https://github.com/kowalej/BlueMuse) for real-time EEG inference
- **Personalization** — online Bayesian calibration adapts to your individual cognitive baseline over sessions
- **Session history** — every run is logged; trend charts and per-session summaries stored locally
- **First-run setup screen** — auto-downloads the dataset from Kaggle with just your API key

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

```bash
python >= 3.10
pip install streamlit torch numpy scipy plotly kaggle
```

### 1. Clone the repo

```bash
git clone https://github.com/NotAverageGitHubUser/Brain-Battery.git
cd Brain-Battery
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the dataset

**Option A — Auto-download (easiest):** Launch the app. If `Data/` is missing, a setup screen appears. Enter your [Kaggle API credentials](https://www.kaggle.com/settings/account) and click **Download Dataset**.

**Option B — Manual:** Download from Kaggle and copy four files into `Data/`:

```
https://www.kaggle.com/datasets/brandon19834/universe-merged-withzero-noasr
```

Required files: `eeg.npy` · `physio.npy` · `label_workload.npy` · `subjects.npy`

### 4. Run

```bash
python -m streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser. Choose **Demo Mode** to explore pre-recorded subjects, or **Live Mode** if you have a Muse headband.

---

## Dataset

The UNIVERSE dataset contains EEG and physiological recordings from 12 subjects performing cognitive tasks at varying mental load levels, labeled with NASA Task Load Index (NASA-TLX) scores.

| Property | Value |
|----------|-------|
| Subjects | 12 |
| EEG channels | 4 (TP9, AF7, AF8, TP10) |
| Physio signals | BVP, HR, EDA |
| Labels | Binary workload (NASA-TLX) |
| Epoch length | 256 samples |
| Validation | Leave-One-Subject-Out (LOSO) |

[**Download on Kaggle →**](https://www.kaggle.com/datasets/brandon19834/universe-merged-withzero-noasr)

---

## Live Mode (Muse Headband)

Live mode streams EEG from a [Muse](https://choosemuse.com/) headband via [BlueMuse](https://github.com/kowalej/BlueMuse) (Windows LSL bridge).

> **Note:** Live mode uses EEG only. Physiological signals (BVP, HR, EDA) are available in Demo mode from the pre-recorded dataset but are not captured by the Muse hardware.

Setup steps:
1. Install [BlueMuse](https://github.com/kowalej/BlueMuse) and start the LSL stream
2. Launch Brain Battery → select **Live Mode**
3. Click **Connect** — the model runs inference on each incoming EEG window

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
