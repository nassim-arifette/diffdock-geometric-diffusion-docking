import pytest

from docking_tools.pilot_summary import aggregate_rows, METRICS


def observation(case, seed, variant, success, evaluated=True):
    return {"complex_id": case, "seed": seed, "variant": variant, "evaluated": evaluated,
            **{k: success for k in METRICS if k != "empty_selection"}, "empty_selection": not evaluated}


def test_failed_cases_stay_in_denominator_and_seeds_are_not_independent_complexes():
    rows = []
    for seed in range(3):
        for variant in ("A_confidence", "B_valid_confidence"):
            rows += [observation("ok", seed, variant, variant == "B_valid_confidence"),
                     observation("failed", seed, variant, False, evaluated=False)]
    result = aggregate_rows(rows)
    assert result["n_complexes"] == 2
    assert result["n_complex_seed_pairs"] == 6
    assert result["variants"]["B_valid_confidence"]["top1_joint_success"] == {"count": 3, "denominator": 6, "rate": 0.5}
    assert result["paired_contrasts"]["B_valid_confidence_minus_A"]["delta_joint_success"] == 0.5


def test_cannot_compare_different_complex_sets_or_double_count_runs():
    rows = [observation("a", 0, "A_confidence", True), observation("b", 0, "B_valid_confidence", True)]
    with pytest.raises(ValueError, match="identical"):
        aggregate_rows(rows)
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_rows(rows + rows)
