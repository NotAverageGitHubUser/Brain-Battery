# Benchmarks — LOSO 2026-04-24

> Generated inline (no Claude API call). Source data: `results/loso_results.json`, `results/research_summary.json`. 24 subjects · 153,669 valid epochs (after dropping 21,184 ambiguous `-1` labels) · CPU.

## Headline

| Model    | Mean acc | Std    | Mean F1 | Mean AUC |
|----------|----------|--------|---------|----------|
| **SANN** | 51.1%    | ±15.5% | 0.625   | 0.563    |
| XGBoost  | 49.9%    | ±8.0%  | 0.534   | —        |

SANN beats XGBoost by **+1.2 pp** mean accuracy with **2× the variance**. Both models are essentially at chance on the LOSO mean.

## The model collapses

**24 / 24 subjects** — at the default 0.5 decision threshold, SANN predicts HIGH cognitive workload for ≥95% of test epochs. The model has no usable threshold without per-subject calibration.

- Mean AUC 0.563 → ranking is mildly above chance.
- 3 subjects with AUC > 0.7 (S01 = 0.84) → the signal exists for some subjects.
- The dashboard's live mode fixes this with **Welford running-stats personalization** (see `personalization.py`); LOSO does not, so these numbers are the no-calibration floor.

## Per-subject SANN

| Subject | n     | acc   | F1    | AUC   |
|---------|-------|-------|-------|-------|
| S00     | 6441  | 52.1% | 0.685 | 0.539 |
| S01     | 7891  | 70.7% | 0.828 | 0.843 |
| S02     | 1910  | 44.6% | 0.616 | 0.556 |
| S03     | 5389  | 60.2% | 0.752 | —     |
| S04     | 3945  | 51.7% | 0.682 | —     |
| S05     | 7510  | 56.2% | 0.720 | —     |
| S06     | 8820  | 68.6% | 0.814 | —     |
| S07     | 1655  | 39.0% | 0.561 | —     |
| S08     | 6496  | 57.0% | 0.727 | —     |
| S09     | 5603  | 62.4% | 0.769 | —     |
| S10     | 5996  | 39.5% | 0.566 | —     |
| S11     | 3551  | 26.8% | 0.410 | 0.525 |
| S12     | 8244  | 59.5% | 0.746 | —     |
| S13     | 10802 | 54.8% | 0.708 | —     |
| S14     | 7300  | 42.5% | 0.597 | —     |
| **S15** | 8119  | 20.2% | 0.336 | 0.444 |
| S16     | 5141  | 60.5% | 0.754 | —     |
| **S17** | 10375 | 71.3% | 0.832 | 0.530 |
| **S18** | 11085 | 79.7% | 0.887 | 0.682 |
| S19     | 1782  | 50.8% | 0.674 | —     |
| S20     | 5810  | 38.4% | 0.554 | —     |
| **S21** | 6891  | 15.6% | 0.269 | 0.450 |
| S22     | 7949  | 46.9% | 0.638 | —     |
| S23     | 4964  | 56.8% | 0.724 | —     |

**Best (top 3):** S18, S17, S01 — all subjects whose actual high-CW prevalence happens to align with the model's collapsed prediction.

**Worst (outliers, ≥1 SD below mean):** S21 (4.4% high-CW prevalence), S15 (sub-chance AUC), S11 (10% high-CW prevalence).

## Confusion matrices

PNGs in `results/plots/`. Pattern across all 24 subjects: the right column dominates — model fires HIGH almost everywhere.

## Recommendations

1. Switch headline metric from 0.5-threshold accuracy to **AUC + per-subject calibrated accuracy**.
2. Implement a LOSO variant that simulates live-mode personalization (warm-start Welford on first 5% of holdout, evaluate on remainder).
3. Inspect S15 and S21 specifically — S15 has sub-chance AUC (signal-quality issue?) and S21 has the most extreme prevalence (4.4% high-CW).
4. Consider reporting balanced accuracy or class-weighted F1 instead of raw accuracy — the per-subject prevalence range (5%–80%) makes raw accuracy misleading.

## Related

- [[Subject_Analysis]] (not generated this session — would require Claude API)
- [[Math_and_Metrics]]
- [[Troubleshooting]]
- `Log 2.md` — full session notes including the eval bugs that had to be fixed first.
