These rules are non-negotiable to maintain dashboard stability and scientific accuracy.

1. **UI Stability (Anti-Jitter):** - Every Plotly chart MUST be wrapped in a container with a defined `min-height` in CSS (e.g., `div[data-testid="stPlotlyChart"] { min-height: 195px; }`).
   - All `st.empty()` placeholders must be declared UNCONDITIONALLY at the top of the `@st.fragment` to lock the DOM height.
2. **Fragment Isolation:** - NEVER call `st.sidebar` inside a function wrapped with `@st.fragment`. This raises a `StreamlitAPIException`.
   - Write debug/state data to `st.session_state._dbg_*` inside the fragment and read it in the sidebar block outside the fragment.
3. **Metric Reversal (The "Workload" Rule):** - The primary metric is "Cognitive Workload" (0% = Rested, 100% = Overloaded). 
   - Variable names must follow the `wl_` prefix (e.g., `wl_hist`, `mean_wl`).
4. **Real-Time Performance:** - Keep inference inside `torch.inference_mode()`.
   - Use `collections.deque` with `maxlen=100` for all rolling histograms to prevent memory leaks.