utf-8
#!/usr/bin/env python3
"""Build a causal AMMT layer-sequence manifest without reading any TIFF pixels.
The manifest defines which manufacturing-layer indices can be used as model
sequence endpoints, and which previous layers are available as causal history.
It prevents a training, validation or test sample from using endpoint layers
from another split.  Guard layers provide a three-layer temporal buffer between
split endpoint ranges.
This script does not open, copy, crop, normalize or modify raw TIFF files.  It
creates only two small text files: a CSV manifest and a JSON split-policy
record.  The CSV is an index; it stores no image pixels.
Default policy
--------------
* Total manufacturing layers: 250
* Causal history K: 4 (endpoint z included)
* Train endpoints: z=4..157 (154 samples)
* Guard layers: z=158..160
* Validation endpoints: z=161..199 (39 samples)
* Guard layers: z=200..202
* Test endpoints: z=203..250 (48 samples)
The 241 usable causal endpoints are allocated as 154/39/48 samples, which is
the closest integer allocation to the approved 64%/16%/20% ratio after
reserving two three-layer guard bands. The first validation endpoint (161) can
use layers 158..161, and the first test endpoint (203) can use layers
200..203. This uses only the preceding guard context and never uses an endpoint
of another split.
Example
-------
cd ~/ammt_project
/usr/local/bin/python3 src/build_causal_manifest.py \
  --output-csv manifests/causal_sequence_manifest.csv \
  --policy-json manifests/split_policy.json
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
@dataclass(frozen=True)
class InclusiveRange:
    """A validated inclusive integer range."""
    start: int
    end: int
    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError(f"range start must be at least 1, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"range end must be >= start, got {self.start}..{self.end}")
    def contains(self, value: int) -> bool:
        return self.start <= value <= self.end
    def values(self) -> range:
        return range(self.start, self.end + 1)
    def as_label(self) -> str:
        return f"{self.start}-{self.end}"
@dataclass(frozen=True)
class SplitSpec:
    """Endpoint range and optional immediately preceding guard range."""
    name: str
    endpoints: InclusiveRange
    preceding_guard: InclusiveRange | None
    def permitted_history_layers(self, sequence_length: int) -> set[int]:
        """Return the historical layers this split may use without leakage.
        Train has no preceding guard. Its earliest legal endpoint therefore
        needs the initial warm-up layers before the first endpoint (e.g. z=1..3
        for K=4 and first endpoint z=4). Validation/test may use only their
        own endpoint range and the immediately preceding guard range.
        """
        allowed = set(self.endpoints.values())
        if self.preceding_guard is not None:
            allowed.update(self.preceding_guard.values())
        else:
            warmup_start = self.endpoints.start - sequence_length + 1
            allowed.update(range(warmup_start, self.endpoints.start))
        return allowed
@dataclass(frozen=True)
class SplitPolicy:
    """Fully explicit causal temporal split used to create the manifest."""
    total_layers: int
    sequence_length: int
    train: SplitSpec
    guard_after_train: InclusiveRange
    validation: SplitSpec
    guard_after_validation: InclusiveRange
    test: SplitSpec
def parse_range(values: list[int], argument_name: str) -> InclusiveRange:
    if len(values) != 2:
        raise ValueError(f"{argument_name} requires exactly two integers")
    return InclusiveRange(start=int(values[0]), end=int(values[1]))
def ranges_overlap(left: InclusiveRange, right: InclusiveRange) -> bool:
    return not (left.end < right.start or right.end < left.start)
def all_policy_ranges(policy: SplitPolicy) -> list[tuple[str, InclusiveRange]]:
    return [
        ("train endpoints", policy.train.endpoints),
        ("guard after train", policy.guard_after_train),
        ("validation endpoints", policy.validation.endpoints),
        ("guard after validation", policy.guard_after_validation),
        ("test endpoints", policy.test.endpoints),
    ]
def validate_policy(policy: SplitPolicy) -> None:
    """Fail early for non-causal, overlapping or out-of-bounds split policies."""
    if policy.sequence_length < 1:
        raise ValueError("sequence length K must be at least 1")
    if policy.total_layers < policy.sequence_length:
        raise ValueError("total layers must be at least the sequence length")
    named_ranges = all_policy_ranges(policy)
    for name, current in named_ranges:
        if current.end > policy.total_layers:
            raise ValueError(f"{name} exceeds total layers {policy.total_layers}: {current.as_label()}")
    for index, (left_name, left_range) in enumerate(named_ranges):
        for right_name, right_range in named_ranges[index + 1 :]:
            if ranges_overlap(left_range, right_range):
                raise ValueError(
                    f"Temporal ranges overlap: {left_name}={left_range.as_label()} and "
                    f"{right_name}={right_range.as_label()}"
                )
    expected_train_end = policy.guard_after_train.start - 1
    expected_validation_start = policy.guard_after_train.end + 1
    expected_validation_end = policy.guard_after_validation.start - 1
    expected_test_start = policy.guard_after_validation.end + 1
    if policy.train.endpoints.end != expected_train_end:
        raise ValueError("train endpoints must end directly before guard-after-train")
    if policy.validation.endpoints.start != expected_validation_start:
        raise ValueError("validation endpoints must start directly after guard-after-train")
    if policy.validation.endpoints.end != expected_validation_end:
        raise ValueError("validation endpoints must end directly before guard-after-validation")
    if policy.test.endpoints.start != expected_test_start:
        raise ValueError("test endpoints must start directly after guard-after-validation")
    first_train_endpoint = policy.train.endpoints.start
    if first_train_endpoint < policy.sequence_length:
        raise ValueError(
            f"first train endpoint z={first_train_endpoint} is too early for K={policy.sequence_length}; "
            f"it must be at least {policy.sequence_length}"
        )
def causal_history(endpoint_z: int, sequence_length: int) -> tuple[int, ...]:
    """Return exactly K layers ending at endpoint_z, with no future-layer access."""
    start_z = endpoint_z - sequence_length + 1
    if start_z < 1:
        raise ValueError(f"endpoint z={endpoint_z} cannot support K={sequence_length}")
    return tuple(range(start_z, endpoint_z + 1))
def build_rows(policy: SplitPolicy) -> list[dict[str, object]]:
    """Create one manifest row per allowed endpoint and validate split isolation."""
    rows: list[dict[str, object]] = []
    for spec in (policy.train, policy.validation, policy.test):
        allowed_history = spec.permitted_history_layers(policy.sequence_length)
        guard_label = "" if spec.preceding_guard is None else spec.preceding_guard.as_label()
        for endpoint_z in spec.endpoints.values():
            history = causal_history(endpoint_z, policy.sequence_length)
            disallowed = [z for z in history if z not in allowed_history]
            if disallowed:
                raise RuntimeError(
                    f"{spec.name} endpoint z={endpoint_z} leaks to another split through history {history}; "
                    f"disallowed layers={disallowed}"
                )
            rows.append(
                {
                    "sample_id": f"{spec.name}_z_{endpoint_z:03d}",
                    "split": spec.name,
                    "endpoint_layer_z": endpoint_z,
                    "history_layer_z": ";".join(str(z) for z in history),
                    "history_start_z": history[0],
                    "history_end_z": history[-1],
                    "sequence_length_k": policy.sequence_length,
                    "preceding_guard_layers": guard_label,
                    "uses_preceding_guard_context": int(
                        spec.preceding_guard is not None and any(spec.preceding_guard.contains(z) for z in history)
                    ),
                    "causal_rule": "history_z <= endpoint_layer_z; no future layer used",
                }
            )
    if not rows:
        raise RuntimeError("policy produced no samples")
    return rows
def rows_by_split(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        split = str(row["split"])
        counts[split] = counts.get(split, 0) + 1
    return counts
def ensure_new(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}. Review it or rerun with --overwrite.")
def write_csv(path: Path, rows: list[dict[str, object]], overwrite: bool) -> None:
    ensure_new(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
def range_dict(value: InclusiveRange | None) -> dict[str, int] | None:
    return None if value is None else asdict(value)
def write_policy_json(path: Path, policy: SplitPolicy, rows: list[dict[str, object]], overwrite: bool) -> None:
    ensure_new(path, overwrite)
    document = {
        "purpose": "Causal layer-sequence index only; no image pixels or labels are stored.",
        "raw_data_policy": "This script never opens or modifies raw TIFF files.",
        "total_manufacturing_layers": policy.total_layers,
        "sequence_length_k": policy.sequence_length,
        "endpoint_ranges": {
            "train": range_dict(policy.train.endpoints),
            "validation": range_dict(policy.validation.endpoints),
            "test": range_dict(policy.test.endpoints),
        },
        "guard_ranges": {
            "after_train": range_dict(policy.guard_after_train),
            "after_validation": range_dict(policy.guard_after_validation),
        },
        "history_rule": (
            "Each sample history is [endpoint_z-K+1, ..., endpoint_z]. Train may use only its endpoint "
            "range plus its initial pre-endpoint warm-up context. Validation/test may use only their own "
            "endpoint layers plus the directly preceding guard layers; endpoint layers of another split "
            "are never used as history."
        ),
        "approved_target_endpoint_ratio": {"train": 0.64, "validation": 0.16, "test": 0.20},
        "sample_count_by_split": rows_by_split(rows),
        "total_sample_count": len(rows),
        "label_policy": "No defect labels, B-A targets, XCT projections or normalization values are assigned here.",
        "next_dependency": "Use the train rows only when estimating stage/LED normalization statistics.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a causal AMMT layer-sequence split manifest without reading TIFFs")
    parser.add_argument("--output-csv", type=Path, required=True, help="CSV path for sample-level sequence indices")
    parser.add_argument("--policy-json", type=Path, required=True, help="JSON path for the split policy and validation record")
    parser.add_argument("--total-layers", type=int, default=250)
    parser.add_argument("--sequence-length", "-k", type=int, default=4)
    parser.add_argument("--train-endpoints", nargs=2, type=int, default=[4, 157], metavar=("START", "END"))
    parser.add_argument("--guard-after-train", nargs=2, type=int, default=[158, 160], metavar=("START", "END"))
    parser.add_argument("--validation-endpoints", nargs=2, type=int, default=[161, 199], metavar=("START", "END"))
    parser.add_argument("--guard-after-validation", nargs=2, type=int, default=[200, 202], metavar=("START", "END"))
    parser.add_argument("--test-endpoints", nargs=2, type=int, default=[203, 250], metavar=("START", "END"))
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing manifest outputs after review")
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    train_endpoints = parse_range(args.train_endpoints, "--train-endpoints")
    guard_after_train = parse_range(args.guard_after_train, "--guard-after-train")
    validation_endpoints = parse_range(args.validation_endpoints, "--validation-endpoints")
    guard_after_validation = parse_range(args.guard_after_validation, "--guard-after-validation")
    test_endpoints = parse_range(args.test_endpoints, "--test-endpoints")
    policy = SplitPolicy(
        total_layers=int(args.total_layers),
        sequence_length=int(args.sequence_length),
        train=SplitSpec("train", train_endpoints, preceding_guard=None),
        guard_after_train=guard_after_train,
        validation=SplitSpec("validation", validation_endpoints, preceding_guard=guard_after_train),
        guard_after_validation=guard_after_validation,
        test=SplitSpec("test", test_endpoints, preceding_guard=guard_after_validation),
    )
    validate_policy(policy)
    output_csv = args.output_csv.resolve()
    policy_json = args.policy_json.resolve()
    if output_csv == policy_json:
        raise ValueError("--output-csv and --policy-json must be different paths")
    ensure_new(output_csv, args.overwrite)
    ensure_new(policy_json, args.overwrite)
    rows = build_rows(policy)
    write_csv(output_csv, rows, args.overwrite)
    write_policy_json(policy_json, policy, rows, args.overwrite)
    counts = rows_by_split(rows)
    print("Causal sequence manifest created. No TIFF file was opened or modified.")
    print(f"- total samples: {len(rows)}")
    print(f"- train: {counts.get('train', 0)} | validation: {counts.get('validation', 0)} | test: {counts.get('test', 0)}")
    print(f"- {output_csv}")
    print(f"- {policy_json}")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
