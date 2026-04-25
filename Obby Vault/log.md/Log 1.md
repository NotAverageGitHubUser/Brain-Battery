> Brain Battery — Implementation Log All entries in reverse chronological order (newest first)

---

## [2026-04-18] feature | Add Claude Code skills for LOSO pipeline

**Branch:** main **Commit:** `e5a0b9b` **Files changed:** `~/.claude/skills/eval-loso/`, `~/.claude/skills/research-agent/`, `~/.claude/skills/interpret-results/`

**Summary:** Created three Claude Code skills to wrap the LOSO evaluation pipeline so it can be invoked as slash commands. Each skill runs pre-flight checks before executing, gives clear error messages if prerequisites are missing, and summarises results on completion.

**Key changes:**
- `/eval-loso`: checks data files + checkpoint, runs `Scripts/eval_loso.py`, summarises accuracy and bottleneck flags, directs user to `/research-agent` next
- `/research-agent`: checks for `loso_results.json`, runs `Scripts/agents/research_agent.py`, reports winner, outliers, false-positive subjects, and bottleneck recommendation
- `/interpret-results`: checks API key + all prerequisites, runs `Scripts/agents/interpreter_agent.py`, previews generated wiki pages, reports token usage

---

## [2026-04-18] feature | Add LOSO eval script, research agent, and Claude interpreter agent

**Branch:** main **Commit:** `e1412eb` **Files changed:** `Scripts/eval_loso.py`, `Scripts/agents/research_agent.py`, `Scripts/agents/interpreter_agent.py`, `requirements.txt`, `.gitignore`

**Summary:** Built a full offline evaluation pipeline for the pre-trained SANN checkpoint. A standalone LOSO runner tests the model against all 24 subjects and compares it to an XGBoost baseline using precomputed spectral features, saving per-subject accuracy, F1, AUC, confusion matrix PNGs, and a bottleneck flag. A Python research agent orchestrates the run and generates a structured summary JSON; a Claude API interpreter agent then reads that JSON plus the worst confusion matrix images and writes formatted Obsidian wiki pages in a single batched API call.

**Key changes:**
- `Scripts/eval_loso.py`: 24-subject LOSO loop — SANN inference (batched, DirectML→CUDA→CPU fallback) + XGBoost on `psd_features.npy`; saves `results/loso_results.json` and `results/plots/subject_XX_*.png`; detects CPU bottleneck when `t_load / t_infer > 1.2`
- `Scripts/agents/research_agent.py`: pure-Python orchestrator — runs eval, ranks subjects, flags outliers (< mean − 1 SD), analyses false-positive rates, outputs `results/research_summary.json`
- `Scripts/agents/interpreter_agent.py`: Claude API agent (`claude-opus-4-7`) — sends aggregated results + up to 6 confusion matrix images in one call; writes `Wiki/Benchmarks.md`, `Wiki/Subject_Analysis.md`, and prepends a log entry; uses prompt caching on system prompt
- `requirements.txt`: added `xgboost`, `matplotlib`, `scikit-learn`, `anthropic`
- `.gitignore`: added `__pycache__/`, `*.pyc`, `results/`

---

## [2026-04-18] feature | Force dark mode via .streamlit/config.toml

**Branch:** main **Commit:** `6029712` **Files changed:** `.streamlit/config.toml`

**Summary:** Added a `.streamlit/config.toml` file that sets `base = "dark"`, ensuring the app always launches in dark mode regardless of the user's OS or browser preference. Prevents the white flash some users see before Streamlit picks up the in-app dark CSS.

**Key changes:**
- `.streamlit/config.toml`: new file with `[theme] base = "dark"`

---

## [2026-04-18] feature | Screenshots grid and Muse setup revamp

**Branch:** main **Commit:** `99ad064` **Files changed:** `dashboard.py`, `docs/index.html`, `docs/assets/screenshot-*.png` (6 files)

**Summary:** Wired 6 real screenshots into the website's screenshot grid and rewrote the Muse connection subpage in the dashboard from a 3-bullet card into a detailed 5-step guide. The setup page now covers hardware requirements, BlueMuse install, Bluetooth pairing, LSL streaming, and in-app connection — plus a "← Back to Live Mode" escape button.

**Key changes:**
- `docs/index.html`: replaced 4 placeholder `.shot` divs with 6 real images (home ×2, demo ×2, live, history) using `%20` URL encoding for filenames with spaces
- `docs/assets/`: added all 6 screenshot PNGs to repo
- `dashboard.py` Muse setup subpage: replaced `.conn-card` + 3 bullets with loop over `_steps` tuples rendered as `.bb-card` number-badge cards (01–05: Hardware Requirements, Install BlueMuse, Pair via Bluetooth, Start Streaming, Connect in Brain Battery)
- Added `← Back to Live Mode` button (`key="muse_back_to_overview"`) that sets `live_subpage = "overview"` and reruns

---

## [2026-04-18] feature | GitHub Pages Website, Full README, and Data Setup Screen

**Branch:** main **Commit:** `3d37046` **Files changed:** `docs/index.html`, `README.md`, `dashboard.py`

**Summary:** Added a full GitHub Pages project website, rewrote the README from scratch, and built a first-run data setup screen in the app. The website matches the dashboard's dark glass aesthetic and includes a full-viewport hero, vertical metrics layout with photo slots, dataset stats, and links to the UNIVERSE paper and original HPI-CH repo. Dataset corrected to 24 subjects throughout.

**Key changes:**
- `docs/index.html`: new GitHub Pages site — sticky glass nav, full-viewport hero with neural network SVG and animated EEG traces, "What it Measures" section as vertical alternating rows each with a photo slot, architecture block, how-it-works steps, screenshot grid, install code blocks, dataset card (24 subjects), featured Nature study card, references; links to Kaggle, HPI-CH GitHub, and Nature paper
- `README.md`: full rewrite — badges, architecture table, install steps, dataset table (24 subjects + published citation), live mode note (EEG-only for Muse), references; links to all three external resources
- `dashboard.py`: first-run setup screen (`render_setup()`) shown when `Data/` files are absent; auto-download path uses Kaggle API credentials; manual path with step-by-step instructions and "check again" button; `check_data_ready()` helper gates all page routing; `KAGGLE_DATASET` / `KAGGLE_URL` / `DATA_FILES` constants added

---

## [2026-04-18] feature | Full Dashboard UI Overhaul

**Branch:** main **Commit:** `f424f9e` **Files changed:** `dashboard.py`

**Summary:** Comprehensive visual and UX overhaul of the entire dashboard. The home page was redesigned with a full-viewport BCI hero, mode-differentiated cards, and plain-English copy. The app page header now shows a large colored mode title instead of the Brain Battery h1. The sidebar was rebuilt as glassmorphic collapsible expanders with forced dark mode.

**Key changes:**
- Home page: full-viewport hero with animated SVG EEG traces, purple demo card and green live card, numbered How it Works bullets (with note that physio signals are Demo-only; Muse is EEG-only in Live mode)
- App header: replaced `🧠 Brain Battery` h1 with large `DEMO MODE` (purple) / `LIVE MUSE MODE` (green) title
- Sidebar: profile avatar removed; Navigate / Playback / Connectivity / Advanced sections converted to `st.expander` dropdowns with glassmorphic curved boxes, backdrop-filter blur, and forced dark mode CSS
- Tabs renamed to plain English: `Live`, `Summary`, `History`; Demo mode shows only `Live` tab
- All emojis removed from labels, buttons, status messages, and popovers for a professional appearance
- Summary tab: plain-English metric names (`Avg Brain Load`, `Avg Tiredness`, `Total Focus Time`), jargon hidden behind nested expander
- History tab: replaced cramped side-column with full-width table (Time / Mode / Brain Load / Tiredness / Focus / Accuracy columns)
- Mode selection removed from sidebar radio — mode is now set exclusively from the home page cards

---

## [2026-04-17] fix | Demo Restart logic and Frame Persistence

**Branch:** main **Files changed:** `dashboard.py`, `Wiki/Troubleshooting.md.md`, `log.md/Log 1.md`

**Summary:** Locked down the fragment state machine so the final frame of a demo run persists on screen, restart is a one-click operation, and Live mode has its own sub-navigation (overview ↔ setup). Added a `live_subpage` session key and reworked Section 18 so the Muse connection flow lives on a dedicated setup page rather than blocking the dashboard.

**Key changes:**

- `st.session_state.live_subpage` default `"overview"` added to `_defaults`
- Section 18 split into two sub-pages — `setup` (SVG placeholder + 1-2-3 steps + Connect button) and `overview` (dashboard with prominent "🎧 Setup Muse Headset" button at top)
- Sidebar: "← Back to Dashboard" button rendered only when `app_mode == Live` **and** `live_subpage == "setup"`
- Subject change handler now sets `_run_complete = False` explicitly in addition to the `_reset_history()` call (defensive, readable)
- Home button clears `live_subpage = "overview"` alongside the other resets
- Fragment completion branch — `return` confirmed absent; `should_run = False` fall-through retained so Phase C render paints the frozen `wl_hist[-1]` / `fatigue_hist[-1]` values
- Muse connect success now sets `live_subpage = "overview"` before `st.rerun()` so the user lands back on the dashboard automatically

**`app_mode` × `live_subpage` transition logic:**

```python
# Home → Live → Overview (default)
if click_home_card_live:
    st.session_state.app_mode     = "🎧 Live — Muse headband"
    st.session_state.live_subpage = "overview"
    st.session_state.page         = "app"

# Overview → Setup
if click_setup_muse_button:
    st.session_state.live_subpage = "setup"
    st.rerun()

# Setup → Overview (via Connect success)
if connect_ok:
    st.session_state.muse_connected = True
    st.session_state.live_subpage   = "overview"
    st.rerun()

# Setup → Overview (via sidebar Back button)
if click_back_to_dashboard:
    st.session_state.live_subpage = "overview"
    st.rerun()

# Any mode → Home (sidebar)
if click_home:
    st.session_state.running       = False
    st.session_state._run_complete = False
    st.session_state.live_subpage  = "overview"
    st.session_state.page          = "home"
    st.rerun()
```

**Fragment state machine (render vs inference split):**

```python
@st.fragment
def live_display():
    # Phase A — unconditional layout + skeleton fill
    ph_hero = st.empty(); ph_fatigue = st.empty(); ...

    # Render defaults = frozen deque tails
    smoothed_wl  = float(st.session_state.wl_hist[-1])
    smoothed_fat = float(st.session_state.fatigue_hist[-1])

    # Phase B — gated inference
    should_run = (st.session_state.running
                  and not st.session_state._run_complete)
    if should_run:
        if frame_idx >= len(f_labels):
            st.session_state._run_complete = True
            st.session_state.running       = False
            should_run = False        # ← fall through, NO return
            HISTORY.append({...})
        else:
            # forward pass → deque appends → frame_idx += 1
            ...

    # Phase C — unconditional render (uses frozen values when paused)
    ph_hero.markdown(make_workload_bar_html(smoothed_wl), ...)
    ...

    # Phase D — loop driver gates on running (not should_run)
    if st.session_state.running:
        time.sleep(speed); st.rerun()
```

**Restart button:**

```python
if st.session_state._run_complete:
    if st.button("↺ Restart Run", type="primary"):
        st.session_state.frame_idx     = 0
        st.session_state._run_complete = False
        st.session_state._final_stats  = {}
        st.session_state.running       = True
        _reset_history()   # clears wl_hist, fatigue_hist, counters, streak
        st.rerun()         # immediate — no fall-through
```

**Wiki:** `Troubleshooting.md.md` gained a "Rendering vs. Inference Split" section that documents the four-phase fragment pattern and the regression-detection checklist.

---

## [2026-04-17] overhaul | Homepage Portal Swap, Pure White Glass UI, and Demo State Fixes

**Branch:** main **Files changed:** `dashboard.py`, `Wiki/Troubleshooting.md.md`, `log.md/Log 1.md`

**Summary:** Full v5.0 overhaul across four phases: state machine hardening, homepage portal swap, pure white glassmorphism UI, and wiki bookkeeping. The dashboard now starts on a landing page; the sidebar is hidden there via CSS injection and restored on the app page.

**Key changes:**

- `DATA_DIR = "Data"`, `MODEL_PATH = "model/best_full_subj_17.pt"` — fixed from Kaggle paths to local
- `st.session_state.page = "home"` default; `render_home()` renders landing page; `st.stop()` prevents app code running
- Hero section: Unsplash brain image, title `letter-spacing: 0.05em`, two glass choice cards (Demo / Live)
- Info section: three columns — How it Works / Goal / Feedback
- Sidebar CSS `display: none` injected only on home page; restored automatically on app page
- Sidebar restructured: **NAVIGATION** (Home button + `_save_session_history()` before nav) / **DEMO SETTINGS** / **LIVE CONNECTIVITY**
- `_save_session_history()` helper appends partial session to history before any navigation away
- Glass cards: `background: rgba(30,35,41,0.6); border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(12px); border-radius: 24px`
- All cyan `#00B3CC` replaced with white `#FFFFFF`; silver `#D1D5DB` used for secondary labels
- Single dynamic button: `"▶ Start"` (disabled when running) or `"↺ Restart Run"` (when `_run_complete`)
- Frame-freeze fix: `return` removed from completion branch; render block always executes using frozen deque values
- `Troubleshooting.md` updated with frame-freeze pattern documentation

**CSS for sidebar hide (home page only):**
```css
div[data-testid="stSidebar"],
div[data-testid="stSidebarCollapsedControl"] { display: none !important; }
```

**Workload formula (retained from v4.2):**
```
Workload = 100 × P(high CW)   — 0% rested, 100% maxed out
Colors: 0–30% #00CC77 / 30–70% #E09000 / 70–100% #E05050
```

---

## [2026-04-16] refactor | Bandwidth to Workload Reversal + Splash Screen

**Branch:** v5.0 **Files changed:** `dashboard.py`, `wiki/Troubleshooting.md`, `wiki/Math_and_Metrics.md`

**Summary:** Complete refactor of the hero metric from Bandwidth (high=good) to Cognitive Workload (high=bad). Added onboarding splash screen with two large mode-selection cards. Sectioned sidebar into HOME / DEMO / LIVE. Advanced settings moved into `st.sidebar.expander`.

**Key changes:**

- `bw_hist` → `wl_hist` (global rename)
- `mean_bw` → `mean_wl` (global rename, backward compat in history chart)
- `personalized_bandwidth()` → `personalized_workload()` (direction reversed)
- `_bw_color()` → `_wl_color()` (palette inverted: low=green, high=red)
- `make_energy_bar_html()` → `make_workload_bar_html()` labels updated
- Pill text: "LOW EFFORT" / "ACTIVE" / "HEAVY LOAD"
- Splash screen: `st.session_state.onboarded = False` → show home cards
- Sidebar sections: HOME (about), DEMO (subject/replay), LIVE (Muse/calibration)
- All Phase 1 state machine fixes from v4.2 retained

**State machine diagram:**

```
[SPLASH] ──select mode──▶ [IDLE]
[IDLE]   ──Start──────▶ [RUNNING] ──end of data──▶ [COMPLETE]
                             ▲                           │
                             └─────Restart Run───────────┘
```

---

## [2026-04-16] bugfix | State Machine Loop Death (v4.2)

**Files changed:** `dashboard.py`

**Summary:** Fixed two bugs where the fragment loop would die after a run completed or a subject was changed in the sidebar.

**Root causes:**

1. Subject change didn't set `running=False` before rerun — stale flag.
2. Restart button didn't call `st.rerun()` immediately — fell through.
3. Loop driver was gated on `should_run` not `running` — missed transition ticks.

**Fix pattern:**

```python
# All state transitions must call st.rerun() immediately
if restart_button_clicked:
    _reset_history()
    st.session_state.running = True
    st.rerun()   # ← immediate, no fall-through

# Loop driver gates on running, not inference success
if st.session_state.running:
    time.sleep(speed); st.rerun()
```

---

## [2026-04-14] feature | Vertical Health Feed Layout (v4.0)

**Files changed:** `dashboard.py`

**Summary:** Complete layout overhaul from multi-column grid to vertical Apple Health-style feed. Each metric gets its own horizontal row: left = visual, right = description. Skeleton loaders added to all placeholders. Tab navigation introduced. History logger (JSON Lines) added. Focus Streak added.

**Anti-jitter solution documented in:** `wiki/Troubleshooting.md`

---

## [2026-04-13] bugfix | st.sidebar inside @st.fragment crash (v3.0)

**Files changed:** `dashboard.py`

**Error:** `StreamlitAPIException: Calling st.sidebar in a function wrapped with st.fragment is not supported.`

**Fix:** Session-state relay pattern. Debug values written to `_dbg_*` keys inside fragment, read by sidebar block outside the fragment.

---

## [2026-04-12] feature | UI Overhaul + Donut Rings (v2.0)

**Files changed:** `dashboard.py`

**Summary:** Replaced Plotly indicator gauge with Neurable-style donut rings. Added DM Mono + DM Sans typography. Muted colour palette (#00B3CC teal). Raw waveforms moved into collapsed st.expander.