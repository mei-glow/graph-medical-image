from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List


DEFAULT_SIZE_BUCKETS = {
    "small": (0.0, 0.05),
    "medium": (0.05, 0.2),
    "large": (0.2, 1.0),
}


def assign_size_bucket(area_ratio: float, buckets: Dict[str, tuple[float, float]] | None = None) -> str:
    buckets = buckets or DEFAULT_SIZE_BUCKETS
    for name, (lower, upper) in buckets.items():
        if lower <= area_ratio < upper:
            return name
    return "large"


def summarize_rows(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}

    numeric_keys = [key for key in rows[0] if key != "id" and key != "size_bucket"]
    summary = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary[key] = sum(values) / len(values)
    summary["num_samples"] = len(rows)
    return summary


def stratify_by_lesion_size(
    per_image_rows: List[Dict[str, float]],
    buckets: Dict[str, tuple[float, float]] | None = None,
) -> Dict[str, Dict[str, float]]:
    buckets = buckets or DEFAULT_SIZE_BUCKETS
    grouped: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for row in per_image_rows:
        bucket = assign_size_bucket(float(row["lesion_area_ratio"]), buckets=buckets)
        enriched = dict(row)
        enriched["size_bucket"] = bucket
        grouped[bucket].append(enriched)

    summary = {}
    for bucket_name in buckets:
        summary[bucket_name] = summarize_rows(grouped.get(bucket_name, []))
    return summary


def compute_domain_gap(
    source_summary: Dict[str, float],
    target_summary: Dict[str, float],
    metrics: Iterable[str] = ("dice", "boundary_dice", "hd95", "iou"),
) -> Dict[str, float]:
    result = {}
    for metric in metrics:
        source_value = source_summary.get(metric)
        target_value = target_summary.get(metric)
        if source_value is None or target_value is None:
            continue
        result[f"{metric}_source"] = source_value
        result[f"{metric}_target"] = target_value
        result[f"{metric}_gap"] = source_value - target_value
    return result
