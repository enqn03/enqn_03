#!/usr/bin/env python3
"""Audit the Top1-Top2 score margin and peak distance distribution for candidate predictions.

This script parses the test coordinate candidates from the A-only Temporal Difference model,
calculates the score margin and spatial distance between the Rank 1 and Rank 2 candidates
for each layer, and simulates how many predictions would be withheld under different
margin safety thresholds (1%, 2%, 5%).
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any


def calculate_euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-json", type=Path, default=Path("outputs/a_only_temporal_difference_v1/test_coordinate_candidates.json"), help="Path to test_coordinate_candidates.json")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/audit_margin_distribution"), help="Directory to save the analysis results")
    args = parser.parse_args()

    if not args.candidates_json.exists():
        raise FileNotFoundError(f"Candidates JSON not found: {args.candidates_json}")

    with open(args.candidates_json, "r") as f:
        data = json.load(f)
    
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("No candidates found in the JSON file.")

    # Group candidates by layer_z
    layer_candidates: dict[int, list[dict[str, Any]]] = {}
    for cand in candidates:
        layer_z = cand["layer_z"]
        if layer_z not in layer_candidates:
            layer_candidates[layer_z] = []
        layer_candidates[layer_z].append(cand)
    
    analysis_results = []
    
    for layer_z, cands in layer_candidates.items():
        # Sort by rank just in case
        cands.sort(key=lambda x: x["rank"])
        
        if len(cands) < 2:
            print(f"Warning: Layer {layer_z} has less than 2 candidates. Skipping.")
            continue
            
        rank1 = cands[0]
        rank2 = cands[1]
        
        if rank1["rank"] != 1 or rank2["rank"] != 2:
            print(f"Warning: Layer {layer_z} missing rank 1 or rank 2. Skipping.")
            continue
            
        margin = rank1["score"] - rank2["score"]
        distance = calculate_euclidean_distance(
            rank1["x_pixel"], rank1["y_pixel"],
            rank2["x_pixel"], rank2["y_pixel"]
        )
        
        analysis_results.append({
            "layer_z": layer_z,
            "rank1_score": rank1["score"],
            "rank2_score": rank2["score"],
            "margin": margin,
            "peak_distance_pixels": distance
        })

    if not analysis_results:
        raise ValueError("No valid layer analyses could be computed.")

    # Calculate statistics
    margins = [res["margin"] for res in analysis_results]
    distances = [res["peak_distance_pixels"] for res in analysis_results]
    
    total_layers = len(analysis_results)
    
    # Margin withholding simulation
    thresholds = [0.01, 0.02, 0.05]
    withhold_counts = {str(t): sum(1 for m in margins if m < t) for t in thresholds}
    withhold_percentages = {str(t): (count / total_layers) * 100 for t, count in withhold_counts.items()}
    
    summary = {
        "purpose": "Simulate margin-based safety withholding policy",
        "total_layers_analyzed": total_layers,
        "margin_stats": {
            "mean": sum(margins) / total_layers,
            "min": min(margins),
            "max": max(margins),
            "median": sorted(margins)[total_layers // 2]
        },
        "distance_stats_pixels": {
            "mean": sum(distances) / total_layers,
            "min": min(distances),
            "max": max(distances),
            "median": sorted(distances)[total_layers // 2]
        },
        "withholding_simulation": {
            t: {
                "threshold_score_diff": float(t),
                "withheld_count": withhold_counts[t],
                "withheld_percentage": round(withhold_percentages[t], 2)
            } for t in map(str, thresholds)
        },
        "layer_details": analysis_results
    }

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / "margin_analysis.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary to stdout
    print("=" * 50)
    print("Margin-based Withholding Audit Summary")
    print("=" * 50)
    print(f"Total layers analyzed: {total_layers}")
    print(f"Average Top1-Top2 Margin: {summary['margin_stats']['mean']:.4f}")
    print(f"Average Peak Distance (pixels): {summary['distance_stats_pixels']['mean']:.2f}")
    print("-" * 50)
    print("Withholding Simulation (If margin < threshold, then withhold):")
    for t_str, data in summary["withholding_simulation"].items():
        print(f"  Threshold {data['threshold_score_diff']:.2f} ({(float(t_str)*100):.0f}%): "
              f"Withheld {data['withheld_count']}/{total_layers} layers "
              f"({data['withheld_percentage']}%)")
    print("=" * 50)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
