# AGENTS.md for Brain Battery Project

## Project Context
We are building a real-time cognitive workload monitor in Streamlit.
- **Main file:** `dashboard.py`
- **Data:** `Data/` folder (ignored by Git)
- **Model:** `model/best_full_subj_17.pt` (ignored by Git)

## Critical Instructions (MUST FOLLOW)
1. **State Management:** ALL session data MUST be stored in `st.session_state`. Never use global variables.
2. **Fragment Rule:** NEVER call `st.sidebar.*` inside the `@st.fragment` function `live_display()`. This raises a `StreamlitAPIException`.
3. **UI Layout:** Do NOT alter the vertical layout scaffolding or CSS classes (e.g., `bb-card`, `pill-green`). The min-height CSS prevents UI jumps.
4. **Testing:** After making changes, do NOT run `streamlit run` directly. The app is already running in my terminal and will hot-reload.

## Current Refactor Tasks
- **Core Metric:** Changing from "Cognitive Bandwidth" to "Cognitive Workload."
  - `Bandwidth = 100 * (1 - P(high CW))` -> `Workload = 100 * P(high CW)`
- **Variable Renames:**
  - `bw_hist` -> `wl_hist`, `mean_bw` -> `mean_wl`
  - `_bw_color` -> `_wl_color`, `_bw_pill` -> `_wl_pill`
- **Color Logic:** The color thresholds for workload are reversed:
  - `0-30%`: Green (`#00CC77`), pill: `"LOW EFFORT"`
  - `30-70%`: Amber (`#E09000`), pill: `"ACTIVE"`
  - `70-100%`: Red (`#E05050`), pill: `"HEAVY LOAD"`
- **Bug Fixes:**
  - Changing the subject in the sidebar MUST reset `st.session_state.running = False` and call `_reset_history()`.
  - The "Restart" button MUST clear all deques, reset `frame_idx = 0`, and set `_run_complete = False`.

## Workflow
- Use `/task` to create a plan for any complex change.
- Before proposing a fix, read the relevant code block and the "Critical Instructions" above.
- Use `git add` and `git commit` with clear, descriptive messages after each successful change.