#!/usr/bin/env python3
"""Generate Table 1 from model predictions regenerated from source imagery.

No historical result/ directory or archived CSV is read.
"""
import argparse
import json
from pathlib import Path
from tess_uav.analysis.temporal_flicker import evaluate_sequence, summarize, write_csv

MODELS = {
    "sam_heuristic": ("mask.png", 29.47),
    "segformer": ("baseline_mask.png", 25.39),
    "segearth": ("segearth_mask.png", 15.73),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions-root", required=True, help="Fresh output directory containing sam_heuristic/, segformer/, segearth/; each has per-frame subdirectories")
    p.add_argument("--output-dir", default="outputs/table1")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--num-frames", type=int, default=20)
    p.add_argument("--ignore-label", type=int, default=0)
    args = p.parse_args()
    root = Path(args.predictions_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for model in args.models:
        mask_name, paper = MODELS[model]
        rows = evaluate_sequence(root/model, mask_name, args.num_frames, args.ignore_label)
        write_csv(output/f"{model}_pairwise.csv", rows)
        summary = summarize(rows)
        summary.update(model=model, target_paper_percent=paper,
                       rounded_matches_paper=round(summary["mean_flicker_percent"], 2) == paper)
        results.append(summary)
        print(f"{model}: {summary['mean_flicker_percent']:.6f}% "
              f"({summary['num_pairs']} pairs); paper {paper:.2f}%")
    write_csv(output/"table1.csv", results)
    (output/"table1.json").write_text(json.dumps(results, indent=2)+"\n")
    print("Saved", output)


if __name__ == "__main__":
    main()
