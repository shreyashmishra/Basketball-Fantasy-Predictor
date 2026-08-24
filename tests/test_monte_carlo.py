import numpy as np

from fantasy.monte_carlo import simulate_season_totals, summarize_simulation


def test_sampler_shape_and_percentiles() -> None:
    draws = simulate_season_totals(10.0, [8, 10, 12], [70, 80, 82], draws=1000, rng=np.random.default_rng(42))
    assert draws.shape == (1000,)
    summary = summarize_simulation(10.0, draws)
    assert summary.p05 <= summary.p25 <= summary.median <= summary.p75 <= summary.p95

