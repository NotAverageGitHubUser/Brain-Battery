"""
research_agent.py — Python-only ML orchestrator for Brain Battery.

Runs eval_loso.py, detects hardware bottlenecks, ranks subjects, and saves
a structured summary ready for the interpreter agent to consume.

No LLM calls — this agent is pure computation.

Usage:
    python scripts/agents/research_agent.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent.parent
RESULTS     = ROOT / "results"
LOSO_JSON   = RESULTS / "loso_results.json"
SUMMARY_OUT = RESULTS / "research_summary.json"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Run the LOSO eval (or load existing results)
# ══════════════════════════════════════════════════════════════════════════════

def ensure_loso_results(force_rerun: bool = False):
    if LOSO_JSON.exists() and not force_rerun:
        print(f"Found existing results at {LOSO_JSON}  (pass --rerun to redo)")
        return
    print("Running LOSO evaluation...")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_loso.py")],
        check=True
    )
    if result.returncode != 0:
        sys.exit("eval_loso.py failed — check the output above.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Analyse results and generate research summary
# ══════════════════════════════════════════════════════════════════════════════

def analyse(data: dict) -> dict:
    summary   = data["summary"]
    subjects  = data["per_subject"]

    # ── Per-subject ranking ───────────────────────────────────────────────────
    sann_rows = [
        {"subject": s["subject"],
         "acc": s["sann"]["acc"],
         "f1":  s["sann"]["f1"],
         "auc": s["sann"]["auc"]}
        for s in subjects if s["sann"] is not None
    ]
    xgb_rows = [
        {"subject": s["subject"],
         "acc": s["xgboost"]["acc"],
         "f1":  s["xgboost"]["f1"],
         "auc": s["xgboost"]["auc"]}
        for s in subjects if s["xgboost"] is not None
    ]

    sann_rows.sort(key=lambda x: x["acc"])
    xgb_rows.sort(key=lambda x: x["acc"])

    # ── Outlier detection (< mean - 1 SD) ────────────────────────────────────
    if sann_rows:
        mean_acc = summary["sann_mean_acc"]
        std_acc  = summary["sann_std_acc"]
        threshold = mean_acc - std_acc
        outliers  = [r for r in sann_rows if r["acc"] < threshold]
    else:
        outliers = []

    # ── Bottleneck report ─────────────────────────────────────────────────────
    bottleneck_subjects = [
        s["subject"]
        for s in subjects
        if s["sann"] and s["sann"].get("bottleneck")
    ]
    bottleneck_flag = len(bottleneck_subjects) > 0

    recommendation = ""
    if bottleneck_flag:
        recommendation = (
            f"CPU bottleneck detected on {len(bottleneck_subjects)} subject(s) "
            f"(S{bottleneck_subjects}). Consider pre-casting tensors to float16 "
            "or increasing DataLoader num_workers if retraining."
        )

    # ── SANN vs XGBoost comparison ────────────────────────────────────────────
    if sann_rows and xgb_rows:
        sann_mean = summary["sann_mean_acc"]
        xgb_mean  = summary["xgb_mean_acc"]
        winner    = "SANN" if sann_mean >= xgb_mean else "XGBoost"
        delta     = abs(sann_mean - xgb_mean)
    else:
        winner, delta = "unknown", 0.0

    # ── Confusion matrix false-positive analysis ──────────────────────────────
    fp_analysis = []
    for s in subjects:
        if s["sann"] is None:
            continue
        cm = s["sann"]["confusion_matrix"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp
        if total == 0:
            continue
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        fp_analysis.append({
            "subject":  s["subject"],
            "fp_rate":  round(fp_rate, 4),
            "fn_rate":  round(fn_rate, 4),
            "note": "High FP: model sees Low→High (natural high-theta?)" if fp_rate > 0.35
                    else "High FN: model sees High→Low (suppressed response?)" if fn_rate > 0.35
                    else "normal"
        })
    fp_analysis.sort(key=lambda x: x["fp_rate"], reverse=True)

    return {
        "model_comparison": {
            "winner": winner,
            "delta_acc": round(delta, 4),
            "sann_mean_acc":  summary["sann_mean_acc"],
            "sann_std_acc":   summary["sann_std_acc"],
            "xgb_mean_acc":   summary["xgb_mean_acc"],
            "xgb_std_acc":    summary["xgb_std_acc"],
        },
        "subject_ranking": {
            "sann_best":   sann_rows[-3:][::-1] if sann_rows else [],
            "sann_worst":  sann_rows[:3],
            "xgb_best":    xgb_rows[-3:][::-1] if xgb_rows else [],
            "xgb_worst":   xgb_rows[:3],
        },
        "outliers": outliers,
        "outlier_threshold": round(threshold, 4) if sann_rows else None,
        "false_positive_analysis": fp_analysis[:5],
        "bottleneck": {
            "detected":        bottleneck_flag,
            "affected_subjects": bottleneck_subjects,
            "recommendation":  recommendation,
        },
        "device": summary.get("device", "unknown"),
        "plots_dir": str(ROOT / "results" / "plots"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    force = "--rerun" in sys.argv
    ensure_loso_results(force_rerun=force)

    with open(LOSO_JSON) as f:
        loso_data = json.load(f)

    print("\nAnalysing results...")
    research_summary = analyse(loso_data)

    with open(SUMMARY_OUT, "w") as f:
        json.dump(research_summary, f, indent=2)

    print(f"\n{'─'*60}")
    mc = research_summary["model_comparison"]
    print(f"Winner:     {mc['winner']}  (Δacc = {mc['delta_acc']:.1%})")
    print(f"SANN acc:   {mc['sann_mean_acc']:.1%} ± {mc['sann_std_acc']:.3f}")
    print(f"XGBoost:    {mc['xgb_mean_acc']:.1%} ± {mc['xgb_std_acc']:.3f}")
    print(f"Outliers:   {len(research_summary['outliers'])} subject(s) below threshold")
    if research_summary["bottleneck"]["detected"]:
        print(f"⚠ Bottleneck: {research_summary['bottleneck']['recommendation']}")
    print(f"\nSummary saved to {SUMMARY_OUT}")
    print("Ready for interpreter agent.")


if __name__ == "__main__":
    main()
