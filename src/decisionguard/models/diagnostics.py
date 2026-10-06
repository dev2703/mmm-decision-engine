"""Shared numerical diagnostic gate, independent of labels and sampler imports."""

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import cast


def acceptable_diagnostics(evidence: Mapping[str, object]) -> bool:
    if evidence.get("sufficient_chains_draws") is not True:
        return False
    for key, minimum, maximum in (
        ("max_rhat", 0, 1.01),
        ("min_ess_bulk", 400, float("inf")),
        ("min_ess_tail", 400, float("inf")),
    ):
        value = evidence.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or not minimum <= float(value) <= maximum
            or (key == "max_rhat" and value == 0)
        ):
            return False
    if any(
        type(evidence.get(key)) is not int or evidence.get(key) != 0
        for key in ("divergences", "maxdepth_reached")
    ):
        return False
    bfmi = evidence.get("bfmi_by_chain")
    if not isinstance(bfmi, Sequence) or isinstance(bfmi, (str, bytes)):
        return False
    values = cast(Sequence[object], bfmi)
    return len(values) >= 2 and all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        and float(value) >= 0.3
        for value in values
    )
