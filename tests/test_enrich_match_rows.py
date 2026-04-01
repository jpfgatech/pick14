import numpy as np

from pick14.rl.pretrain_curriculum import (
    enrich_match_rows_scoring_legal,
    summarize_match_a_t_distribution,
)


def _row(a_t: float, pass_row: int = 2) -> tuple[dict[str, np.ndarray], float]:
    mm = np.zeros((pass_row + 1, 4), dtype=np.int8)
    mm[0, 0] = 1
    obs = {"mask_match": mm}
    return obs, float(a_t)


def test_enrich_boosts_rare_target():
    rows = [_row(3.0), _row(3.0), _row(12.0)]
    out, meta = enrich_match_rows_scoring_legal(
        rows,
        pass_row=2,
        boost_values=[12.0],
        min_count_boost=5,
        min_per_int_bin=0,
        seed=0,
    )
    assert meta["n_eligible_base"] == 3
    assert meta["n_after"] == 3 + 4
    twelves = sum(1 for _o, t in out if int(round(t)) == 12)
    assert twelves == 5


def test_summarize_bins():
    rows = [_row(0.0), _row(0.0), _row(3.0)]
    s = summarize_match_a_t_distribution(rows, pass_row=2)
    assert s["n_eligible"] == 3
    assert s["bins_int"][0] == 2
    assert s["bins_int"][3] == 1
