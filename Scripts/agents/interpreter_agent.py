"""
interpreter_agent.py — Claude API agent for Brain Battery result interpretation.

Reads research_summary.json + confusion matrix PNGs, sends them to Claude
in one batched call, and writes formatted wiki pages to the Obsidian vault.

Requires: ANTHROPIC_API_KEY environment variable.

Usage:
    python scripts/agents/interpreter_agent.py
"""

import base64
import json
import os
import sys
from datetime import date
from pathlib import Path

import anthropic

ROOT         = Path(__file__).resolve().parent.parent.parent
RESULTS      = ROOT / "results"
PLOTS_DIR    = RESULTS / "plots"
SUMMARY_JSON = RESULTS / "research_summary.json"
LOSO_JSON    = RESULTS / "loso_results.json"
VAULT        = ROOT.parent / "Obby Vault"
WIKI_DIR     = VAULT / "Wiki"
LOG_FILE     = VAULT / "log.md" / "Log 1.md"

MODEL = "claude-opus-4-7"
TODAY = date.today().isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Missing: {path}\nRun research_agent.py first.")
    with open(path) as f:
        return json.load(f)


def encode_image(path: str) -> dict:
    """Return an Anthropic image content block from a PNG file path."""
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def pick_confusion_images(summary: dict, loso: dict, max_images: int = 6) -> list[dict]:
    """Select the most informative confusion matrix images to send to Claude."""
    outlier_subjects  = {r["subject"] for r in summary.get("outliers", [])}
    worst_subjects    = {r["subject"] for r in summary["subject_ranking"]["sann_worst"]}
    priority_subjects = outlier_subjects | worst_subjects

    content_blocks = []
    seen = set()

    # Priority: outliers and worst subjects first
    for s in summary["per_subject"] if "per_subject" in summary else loso["per_subject"]:
        subj = s["subject"] if isinstance(s, dict) else s
        pass

    for entry in loso["per_subject"]:
        subj = entry["subject"]
        if subj in priority_subjects and subj not in seen:
            for model_tag in ("sann", "xgboost"):
                result = entry.get(model_tag)
                if result and result.get("confusion_plot"):
                    p = Path(result["confusion_plot"])
                    if p.exists() and len(content_blocks) < max_images:
                        content_blocks.append({"type": "text",
                                               "text": f"Confusion matrix — Subject S{subj:02d} ({model_tag.upper()}):"})
                        content_blocks.append(encode_image(str(p)))
                        seen.add(subj)

    # Fill remaining slots with best/worst contrast
    for entry in loso["per_subject"]:
        subj = entry["subject"]
        if subj not in seen and entry.get("sann") and entry["sann"].get("confusion_plot"):
            p = Path(entry["sann"]["confusion_plot"])
            if p.exists() and len(content_blocks) < max_images * 2:
                content_blocks.append({"type": "text",
                                       "text": f"Confusion matrix — Subject S{subj:02d} (SANN):"})
                content_blocks.append(encode_image(str(p)))
                seen.add(subj)

    return content_blocks


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a neuroscience research analyst writing entries for a technical research wiki.
Your output is Obsidian-flavored markdown. Follow these rules exactly:

1. Use ## for section headers, ### for sub-sections.
2. Cite sources as [[raw/Author_Year.pdf]] (Obsidian wikilink style).
3. Include a markdown table wherever comparing numbers.
4. Be concise but precise — no filler sentences.
5. When discussing false positives (model predicts High CW when actual is Low CW),
   consider the theta-wave hypothesis: some individuals have naturally elevated
   theta (4–8 Hz) activity that mimics cognitive workload signatures [[raw/Klimesch_1999.pdf]].
6. Format numbers as percentages with one decimal place.
"""

def build_user_message(summary: dict, loso: dict, image_blocks: list) -> list:
    mc   = summary["model_comparison"]
    rank = summary["subject_ranking"]
    fp   = summary.get("false_positive_analysis", [])

    results_text = f"""
## LOSO Evaluation Results — Brain Battery

### Model Comparison
| Metric | SANN (Multimodal) | XGBoost (Spectral only) |
|--------|-------------------|------------------------|
| Mean Accuracy | {mc['sann_mean_acc']:.1%} ± {mc['sann_std_acc']:.3f} | {mc['xgb_mean_acc']:.1%} ± {mc['xgb_std_acc']:.3f} |
| Winner | {"✓" if mc['winner'] == 'SANN' else ""} | {"✓" if mc['winner'] == 'XGBoost' else ""} |
| Delta | {mc['delta_acc']:.1%} advantage to {mc['winner']} | |

### Subject Rankings (SANN)
Best 3: {rank['sann_best']}
Worst 3: {rank['sann_worst']}

### Outliers (below mean − 1 SD threshold of {summary.get('outlier_threshold', 'N/A'):.1%})
{json.dumps(summary['outliers'], indent=2)}

### False Positive Analysis (top 5 by FP rate)
{json.dumps(fp, indent=2)}

### Hardware
Device: {summary['device']}
Bottleneck detected: {summary['bottleneck']['detected']}
{summary['bottleneck']['recommendation']}

### Full per-subject SANN results
{json.dumps([{
    'subject': s['subject'],
    'acc': s['sann']['acc'] if s['sann'] else None,
    'f1': s['sann']['f1'] if s['sann'] else None,
    'auc': s['sann']['auc'] if s['sann'] else None,
} for s in loso['per_subject']], indent=2)}
"""

    user_content = [{"type": "text", "text": results_text.strip()}]
    user_content.extend(image_blocks)
    user_content.append({"type": "text", "text": """
Please write the following three wiki pages based on the data and confusion matrices above:

---
PAGE 1: Benchmarks.md
A comparison table of SANN vs XGBoost across all subjects. Include mean ± std for
Accuracy, F1, and AUC. Add a "Takeaway" paragraph explaining which model is better and why.

---
PAGE 2: Subject_Analysis.md
For each outlier subject, write a paragraph explaining likely causes of poor performance.
Use the confusion matrices (especially false positive rates) to hypothesize whether the
issue is a high-theta individual, label noise, or class imbalance.
Include a ranked table of all subjects by SANN accuracy.

---
PAGE 3: Log entry (markdown only, no header)
A single log entry block in this format:
## [{today}] research | LOSO Cross-Validation & GPU Optimization
**Branch:** main **Commit:** `(pending)` **Files changed:** `results/loso_results.json`, `results/plots/`
**Summary:** <2-3 sentences>
**Key changes:**
- <bullet 1>
- <bullet 2>

Separate each page with exactly: ===PAGE_BREAK===
""".replace("{today}", TODAY)})

    return user_content


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY environment variable first.\n"
                 "  Windows: set ANTHROPIC_API_KEY=sk-ant-...")

    summary = load_json(SUMMARY_JSON)
    loso    = load_json(LOSO_JSON)

    print("Selecting confusion matrix images...")
    image_blocks = pick_confusion_images(summary, loso)
    print(f"  Sending {len([b for b in image_blocks if b['type'] == 'image'])} images to Claude")

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Calling {MODEL}...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=[
            {"type": "text", "text": SYSTEM_PROMPT,
             "cache_control": {"type": "ephemeral"}}  # cache system prompt
        ],
        messages=[{"role": "user", "content": build_user_message(summary, loso, image_blocks)}],
    )

    raw = response.content[0].text
    pages = [p.strip() for p in raw.split("===PAGE_BREAK===")]

    if len(pages) < 3:
        print("Warning: Claude returned fewer than 3 pages. Saving raw output.")
        (RESULTS / "interpreter_raw.md").write_text(raw, encoding="utf-8")
        sys.exit(1)

    benchmarks_md, subject_md, log_entry = pages[0], pages[1], pages[2]

    # ── Write wiki pages ──────────────────────────────────────────────────────
    WIKI_DIR.mkdir(parents=True, exist_ok=True)

    bench_path = WIKI_DIR / "Benchmarks.md"
    bench_path.write_text(benchmarks_md, encoding="utf-8")
    print(f"Written: {bench_path}")

    subj_path = WIKI_DIR / "Subject_Analysis.md"
    subj_path.write_text(subject_md, encoding="utf-8")
    print(f"Written: {subj_path}")

    # ── Prepend log entry ─────────────────────────────────────────────────────
    if LOG_FILE.exists():
        existing = LOG_FILE.read_text(encoding="utf-8")
        header_line = existing.split("\n")[0]
        rest        = "\n".join(existing.split("\n")[1:])
        new_log     = f"{header_line}\n\n{log_entry}\n\n---\n{rest.lstrip()}"
        LOG_FILE.write_text(new_log, encoding="utf-8")
        print(f"Log updated: {LOG_FILE}")
    else:
        print(f"Log file not found at {LOG_FILE} — log entry saved to results/log_entry.md")
        (RESULTS / "log_entry.md").write_text(log_entry, encoding="utf-8")

    # ── Usage report ──────────────────────────────────────────────────────────
    usage = response.usage
    print(f"\nTokens — in: {usage.input_tokens}  out: {usage.output_tokens}")
    print("Done.")


if __name__ == "__main__":
    main()
