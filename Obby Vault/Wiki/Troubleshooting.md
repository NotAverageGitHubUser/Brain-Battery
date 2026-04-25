> LLM Wiki — Brain Battery Dashboard Last updated: 2026-04-16

---

## 🚨 CRITICAL GUARDRAIL: Streamlit Screen Jitter / Flash

**Status:** SOLVED (v4.0+, retained through all versions) **Severity:** High — visible to demo judges, breaks the "polished product" perception

### Symptom

The dashboard visibly flashes or jumps vertically every 2 seconds when the fragment reruns. Charts appear to collapse briefly before re-rendering.

### Root Cause

When `st.plotly_chart()` re-renders inside `@st.fragment`, Streamlit briefly removes the old DOM node before inserting the new one. If no minimum height is reserved for the container, the page height collapses momentarily, which the browser interprets as a scroll-anchor violation and triggers a visible upward jump.

### The Fix (DO NOT REMOVE ANY OF THESE)

**1. CSS `min-height` on Plotly containers:**

```css
/* In the global <style> block — must remain in every version */
div[data-testid="stPlotlyChart"] { min-height: 195px; }
div[data-testid="column"]        { padding: 0 5px !important; }
```

This locks the minimum DOM height so the page never collapses between renders. `195px` matches the tallest chart (ring gauge = 195px). If you add a taller chart, increase this value.

**2. Unconditional layout scaffold at the top of `@st.fragment`:**

```python
# WRONG — causes jitter:
if st.session_state.running:
    r1l, r1r = st.columns([2, 1])
    with r1l:
        ph_hero = st.empty()

# CORRECT — always declare placeholders first:
r1l, r1r = st.columns([2, 1])
with r1l:
    ph_hero = st.empty()   # declared unconditionally
# ... then check running state ...
if st.session_state.running:
    ph_hero.markdown(make_workload_bar_html(...), ...)
```

All `st.columns()`, `st.empty()`, and `st.expander()` calls must appear **before** any `if` guard. This ensures DOM nodes exist on every tick.

**3. Skeleton loaders fill placeholders immediately:**

```python
# After declaring all ph_* placeholders, fill with skeleton HTML:
ph_hero.markdown(f'<div style="min-height:110px">{_skel_bar(110)}</div>',
                 unsafe_allow_html=True)
```

Skeleton HTML fills the vertical space with a pulsing grey block. Real content overwrites it on the next inference tick. The `@keyframes skeletonPulse` animation makes the wait feel intentional.

**4. `html { overflow-anchor: none; }` in global CSS:** Prevents the browser from scrolling to keep a "visible" element stable during DOM updates, which was causing the page to jump upward.

### If a future AI tries to "simplify" by removing placeholders or skeletons:

**STOP.** Check this file. The placeholders are load-bearing. Removing them causes screen jitter that is immediately visible in demos. The `_skel_bar()`, `_skel_ring()`, `_skel_text_block()` helpers must stay.

---

## 🚨 CRITICAL GUARDRAIL: st.sidebar.* Inside @st.fragment

**Status:** SOLVED (v3.0+) **Error:** `StreamlitAPIException: Calling st.sidebar in a function wrapped with st.fragment is not supported.`

### The Fix

```python
# WRONG — crashes at runtime:
@st.fragment
def live_display():
    st.sidebar.caption(f"P(CW) = {cw_prob:.3f}")   # ← CRASH

# CORRECT — use session_state relay:
@st.fragment
def live_display():
    # Write debug values to session state
    st.session_state._dbg_cw = cw_prob   # ← safe inside fragment

# Then, OUTSIDE the fragment (in section 13/sidebar):
if st.session_state.get("_dbg_frame", 0) > 0:
    st.sidebar.caption(f"P(CW)={st.session_state._dbg_cw:.3f}")  # ← safe
```

Zero `st.sidebar.*` calls anywhere inside `live_display()`.

---

## 🚨 CRITICAL GUARDRAIL: @st.cache_resource / @st.cache_data + st.* calls

**Status:** DOCUMENTED (all versions) **Error:** `StreamlitAPIException` when a cached function replays st.* output.

### The Rule

All `@st.cache_resource` and `@st.cache_data` functions must contain **zero** `st.*` calls. Return diagnostics as plain dicts and write them to the sidebar _outside_ the cached function.

---

## Frame-Freeze Pattern: Persistent Display After Run Ends

**Status:** SOLVED (v5.0+)

### Symptom

After the demo dataset is exhausted, the dashboard goes blank or shows only the completion banner — gauges and charts reset to zero instead of holding the final frame values.

### Root Cause

The original completion branch contained a `return` statement after setting `_run_complete = True`. This exited the fragment before the render block, so placeholders were never filled with the final values — they stayed as skeleton loaders.

### The Fix

Remove the `return` from the completion branch. Let execution fall through to the render block. The render block reads from `st.session_state.wl_hist[-1]` and `st.session_state.fatigue_hist[-1]` (frozen values), so the final frame is displayed persistently.

```python
# WRONG — exits before render, leaves skeletons on screen:
if i >= len(f_labels):
    st.session_state._run_complete = True
    st.session_state.running = False
    HISTORY.append({...})
    return   # ← DO NOT ADD THIS

# CORRECT — fall through to render with should_run=False:
if i >= len(f_labels):
    st.session_state._run_complete = True
    st.session_state.running = False
    HISTORY.append({...})
    should_run = False
    # Execution continues → render block uses frozen wl_hist[-1] values
```

The `should_run = False` flag prevents the inference block from running, but the render block below is **not gated on `should_run`** — it always executes, using the last appended values from the deques.

---

## Rendering vs. Inference Split (Frame Persistence in Fragments)

**Status:** SOLVED (v5.1+)

### Symptom

The final frame of a demo run disappears — gauges, rings, and charts collapse back to skeletons the moment the dataset is exhausted. Restarting the session also fails because the fragment exits before re-rendering.

### Root Cause

Mixing **inference** (model forward pass, deque appends, frame advance) with **rendering** (placeholder `.markdown()` / `.plotly_chart()` calls) inside a single `if st.session_state.running:` block ties visual state to inference state. When inference stops, rendering stops too, and the placeholders revert to their skeleton defaults.

`@st.fragment` reruns are cheap, but every rerun must repaint every placeholder — Streamlit does not "remember" what a placeholder held on the previous run. If the render branch is skipped, the DOM node gets re-filled with whatever was placed into it first (skeletons).

### The Fix — Split the Fragment Into Two Phases

```python
@st.fragment
def live_display():
    # ── Phase A: LAYOUT + SKELETONS (unconditional) ──
    r1l, r1r = st.columns([2, 1])
    ph_hero  = st.empty()
    # … declare every ph_* placeholder, fill with skeleton HTML …

    # ── Render-variable defaults (used if not running) ──
    smoothed_wl  = float(st.session_state.wl_hist[-1])
    smoothed_fat = float(st.session_state.fatigue_hist[-1])
    attn_vals    = np.array([0.33, 0.33, 0.34])

    # ── Phase B: INFERENCE (gated) ──
    should_run = (
        st.session_state.running
        and not st.session_state._run_complete
        and frag_model is not None
    )
    if should_run:
        i = st.session_state.frame_idx
        if i >= len(f_labels):
            # Dataset exhausted — commit stats, flip flags, DO NOT return.
            st.session_state._run_complete = True
            st.session_state.running       = False
            should_run = False
            # Fall through to Phase C with frozen wl_hist[-1] values.
        else:
            # … model forward pass, deque appends, frame advance …

    # ── Phase C: RENDER (unconditional) ──
    ph_hero.markdown(make_workload_bar_html(smoothed_wl), unsafe_allow_html=True)
    ph_fatigue.plotly_chart(make_ring(smoothed_fat, …), key="fatigue_ring")
    # … every placeholder repainted every tick …

    # ── Phase D: LOOP DRIVER ──
    if st.session_state.running:
        time.sleep(speed); st.rerun()
```

### Key Invariants

1. **Phase A** declares every placeholder *unconditionally* — no `if` guards around `st.columns`, `st.empty`, or `st.expander`. This protects against the jitter bug and ensures DOM nodes always exist.
2. **Phase B** may set `should_run = False` and fall through; it must **never** `return` or `st.stop()` from the completion branch. The `return` from the Muse-buffering and unstable-signal branches is an intentional exception — those are mid-run transient states, not run-completion.
3. **Phase C** reads render variables that default to `wl_hist[-1]` / `fatigue_hist[-1]`. When inference is paused, these frozen values flow into every placeholder, so the final frame persists indefinitely.
4. **Phase D** gates the loop on `running`, not `should_run` — the loop must still drive reruns while Phase B is in a transient skip state (buffering, unstable signal).

### How to Spot a Regression

If you see gauges go blank after "Run Complete", search for any of:
- `return` inside the `if should_run:` / `if st.session_state.running:` blocks of the fragment
- `st.stop()` inside the fragment
- A `ph_*.markdown()` or `ph_*.plotly_chart()` call nested under an `if` that depends on `running` or `should_run`

All three are the same bug wearing different hats.

---

## State Machine Bug: Loop Dies After Run Complete or Subject Change

**Status:** SOLVED (v4.2+)

### Symptom

After a demo run completes, clicking "↺ Restart Run" does nothing. Changing subjects in the sidebar also fails to start a new run.

### Root Cause

1. `running=True` was set but the fragment wasn't forced to rerun immediately. The next fragment rerun saw `_run_complete=True` with stale history deques and stopped without inference.
2. The loop driver `if should_run: st.rerun()` was gated on inference success, so any tick where acquisition was skipped (transition tick) killed the loop.

### The Fix

```python
# RESTART BUTTON — must call st.rerun() immediately, not rely on fall-through:
if st.button("↺ Restart Run", type="primary"):
    st.session_state.frame_idx     = 0
    st.session_state._run_complete = False
    st.session_state._final_stats  = {}
    st.session_state.running       = True
    _reset_history()   # wipes wl_hist, fatigue_hist, counters, streak
    st.rerun()         # ← MUST be here

# SUBJECT CHANGE — must set running=False and clear history:
if subj_sel != st.session_state.subject_idx:
    st.session_state.subject_idx = subj_sel
    st.session_state.frame_idx   = 0
    st.session_state.running     = False    # ← explicit
    st.session_state._eeg_data   = None
    st.session_state._loaded_subject = -1
    _reset_history()                        # ← required
    st.rerun()

# LOOP DRIVER — gate on running, not on inference success:
# WRONG:
if should_run:
    time.sleep(speed); st.rerun()

# CORRECT:
if st.session_state.running:    # ← always rerun when running
    time.sleep(speed); st.rerun()
```