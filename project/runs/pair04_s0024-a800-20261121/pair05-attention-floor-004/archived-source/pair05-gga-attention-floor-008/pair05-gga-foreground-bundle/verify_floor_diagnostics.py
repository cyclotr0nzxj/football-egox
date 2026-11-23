"""Require informative attention-floor samples in the isolated Pair05 result."""

import json
import math


SAMPLE_PREFIX = b"PAIR05_FLOOR_SAMPLE "
EXPECTED_CALLS = (1, 2, 2000, 2001, 3999, 4000)
TARGETS = {"F12:1", "F12:2", "F12:6", "F24:1", "F24:2", "F24:4", "F24:6"}
FLOOR = 0.25


def probability(value):
    return type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1.0001


def verify_samples(payload, *, expected_calls=EXPECTED_CALLS):
    samples = [
        json.loads(line[len(SAMPLE_PREFIX):])
        for entry in payload.splitlines()
        if (line := entry.strip()).startswith(SAMPLE_PREFIX)
    ]
    if tuple(sample["call"] for sample in samples) != tuple(expected_calls):
        raise ValueError("Missing, duplicate, or out-of-order Pair05 attention-floor samples.")
    for sample in samples:
        if (sample["floor"] != FLOOR or sample["heads"] != 40 or sample["branches"] != 53
                or not probability(sample["minimum_after"]) or sample["minimum_after"] < 0.245
                or not probability(sample["maximum_before"])
                or type(sample["heads_branches_raised"]) is not int
                or not 0 <= sample["heads_branches_raised"] <= 40 * 53
                or set(sample["by_target"]) != TARGETS):
            raise ValueError("Invalid Pair05 measured attention-floor layout or mass.")
        for group in sample["by_target"].values():
            prior = [group[f"before_{stat}"] for stat in ("min", "mean", "max")]
            adjusted = [group[f"after_{stat}"] for stat in ("min", "mean", "max")]
            norms = [group[key] for key in ("exo_output_rms", "routed_output_rms")]
            if (not all(probability(value) for value in prior + adjusted)
                    or prior != sorted(prior) or adjusted != sorted(adjusted)
                    or not all(type(value) in (float, int) and math.isfinite(value) and value >= 0 for value in norms)
                    or adjusted[0] < 0.245 or adjusted[1] + 0.0001 < prior[1]):
                raise ValueError("Invalid Pair05 per-object before/after attention mass.")
    return {"floor": FLOOR, "sampled_calls": list(expected_calls), "samples": samples}
