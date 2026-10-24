# Manuscript value verification

This is a **diagnostic**, not a procedure that forces metrics to match historical outputs.
It compares generated outputs to a checked-in reference manifest at the precision displayed in that reference.

```bash
python scripts/verify_manuscript.py --output-root outputs
```

Output: `outputs/manuscript_verification.json`, with PASS, DIFFER, MISSING and UNSUPPORTED for each value. Use `--strict` only once every expected value is actually reproduced; otherwise it exits nonzero by design.

**Important:** `configs/arxiv_v1_tables.json` is transcribed from the public **arXiv:2608.28665v1 (23 Aug 2026)**, not independently confirmed against the accepted ICVGIP camera-ready. Replace it with a verified copy derived from the final accepted PDF (`--reference configs/accepted_tables.json`) before making camera-ready reproducibility claims.

Scope: Tables 1–4 of the preprint. Table 1 raw SAM proposals are listed in the paper but missing from the current three-model evaluator, and are intentionally reported MISSING. Class-wise entropy is not available unconditioned from the existing classwise CSV, so the checker reports UNSUPPORTED rather than substitute a conditioned estimate. Some 3D numbers will differ from the preprint; this is a meaningful discrepancy, not a tool failure. The narrative cross-backbone and sensitivity numbers also require their own manifest and executable pipeline checks before claiming full-paper reproducibility.
