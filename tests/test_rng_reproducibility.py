"""RNG reproducibility, seed-type equivalence, thread safety, and
non-pollution of ``numpy.random`` global state.

These tests pin the behaviour contract that PR #178 established when the
code base moved from the legacy ``numpy.random.seed`` / ``np.random.randint``
(global-state) pattern to per-call ``numpy.random.Generator`` instances.

The key test here --- ``test_max_min_sampling_thread_isolation`` --- would
have failed under the prior design because two threads simultaneously
calling ``np.random.seed(...)`` race on the global state; whichever thread
re-seeds last clobbers the other. The current design gives each call its
own ``Generator`` so concurrent callers cannot interfere.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from palantir.core import _max_min_sampling, run_palantir


def _make_df(n_cells: int = 80, n_dims: int = 6) -> pd.DataFrame:
    """Deterministic synthetic DataFrame for sampling tests.

    We build it from a fixed seed so the test inputs themselves don't
    depend on global RNG state or on any other test's ordering.
    """
    rng = np.random.default_rng(12345)
    return pd.DataFrame(
        rng.random((n_cells, n_dims)),
        columns=[f"DC_{i}" for i in range(n_dims)],
        index=[f"cell_{i}" for i in range(n_cells)],
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_max_min_sampling_int_seed_determinism():
    """Same integer seed must yield bitwise-identical waypoint indices."""
    df = _make_df()
    out_a = _max_min_sampling(df, num_waypoints=30, seed=42)
    out_b = _max_min_sampling(df, num_waypoints=30, seed=42)
    assert list(out_a) == list(out_b)


def test_max_min_sampling_different_seeds_differ():
    """Different seeds should (with overwhelming probability) diverge.

    Guards against accidental seed-discarding regressions --- e.g., if
    someone refactors ``default_rng(seed)`` to ``default_rng()``.
    """
    df = _make_df()
    out_a = _max_min_sampling(df, num_waypoints=30, seed=42)
    out_b = _max_min_sampling(df, num_waypoints=30, seed=7)
    assert list(out_a) != list(out_b)


# ---------------------------------------------------------------------------
# Int <-> Generator equivalence
# ---------------------------------------------------------------------------


def test_max_min_sampling_int_generator_equivalence():
    """``seed=42`` and ``seed=default_rng(42)`` must be equivalent.

    ``numpy.random.default_rng`` returns the passed-in Generator unchanged,
    so constructing one from the same int and passing it in should produce
    the same sequence.
    """
    df = _make_df()
    out_from_int = _max_min_sampling(df, num_waypoints=30, seed=42)
    out_from_gen = _max_min_sampling(
        df, num_waypoints=30, seed=np.random.default_rng(42)
    )
    assert list(out_from_int) == list(out_from_gen)


def test_max_min_sampling_seed_sequence_equivalence():
    """A ``SeedSequence(42)`` should be equivalent to the int ``42``."""
    df = _make_df()
    out_from_int = _max_min_sampling(df, num_waypoints=30, seed=42)
    out_from_ss = _max_min_sampling(
        df, num_waypoints=30, seed=np.random.SeedSequence(42)
    )
    assert list(out_from_int) == list(out_from_ss)


# ---------------------------------------------------------------------------
# Thread safety --- the "would have failed under prior design" test
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:.*is too small.*:UserWarning")
def test_max_min_sampling_thread_isolation():
    """Two concurrent calls with different seeds must not interfere.

    The previous design did::

        if seed is not None:
            np.random.seed(seed)
        ...
        current_wp = np.random.randint(N)

    Under that design, two threads entering ``_max_min_sampling`` with
    different seeds race on the global ``numpy.random`` state: whichever
    thread's ``np.random.seed`` call lands last dictates the RNG state
    for both threads' subsequent ``randint`` draws. The parallel outputs
    then drift away from their sequential baselines.

    The new design assigns each call its own ``Generator``, so the
    parallel and sequential outputs must match exactly.
    """
    df = _make_df()
    seed_a, seed_b = 101, 202

    # Sequential baselines.
    seq_a = list(_max_min_sampling(df, num_waypoints=40, seed=seed_a))
    seq_b = list(_max_min_sampling(df, num_waypoints=40, seed=seed_b))

    # Run many rounds in parallel to make interleaving likely. Under the
    # legacy global-state design this loop would almost certainly surface
    # a mismatch; under the Generator design it must always match.
    for _ in range(20):
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(
                _max_min_sampling, df, num_waypoints=40, seed=seed_a
            )
            fut_b = pool.submit(
                _max_min_sampling, df, num_waypoints=40, seed=seed_b
            )
            par_a = list(fut_a.result())
            par_b = list(fut_b.result())
        assert par_a == seq_a
        assert par_b == seq_b


# ---------------------------------------------------------------------------
# No pollution of numpy's global RNG state
# ---------------------------------------------------------------------------


def test_max_min_sampling_does_not_touch_global_rng():
    """Seeded call must not mutate the legacy global ``numpy.random`` state.

    This would have failed under the prior design, which called
    ``np.random.seed(seed)`` and ``np.random.randint(...)`` directly on
    the process-wide singleton.
    """
    # Anchor the global state to a known point, then snapshot it.
    np.random.seed(999)
    before = np.random.get_state()

    df = _make_df()
    _max_min_sampling(df, num_waypoints=30, seed=42)

    after = np.random.get_state()
    # ``get_state`` returns a tuple; element 1 is the 624-uint32 MT state.
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated.")
@pytest.mark.filterwarnings(
    "ignore:Changing the sparsity structure of a csr_matrix is expensive."
)
def test_run_palantir_does_not_touch_global_rng(mock_data):
    """Same non-pollution guarantee at the public ``run_palantir`` entry point."""
    np.random.seed(999)
    before = np.random.get_state()

    run_palantir(mock_data, "cell_0", seed=42)

    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


# ---------------------------------------------------------------------------
# End-to-end: same seed at the run_palantir entry point -> same pseudotime
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated.")
@pytest.mark.filterwarnings(
    "ignore:Changing the sparsity structure of a csr_matrix is expensive."
)
def test_run_palantir_int_seed_determinism(mock_data):
    """``run_palantir`` with the same int seed is end-to-end deterministic."""
    r1 = run_palantir(mock_data, "cell_0", seed=42)
    r2 = run_palantir(mock_data, "cell_0", seed=42)
    np.testing.assert_array_equal(r1.pseudotime.values, r2.pseudotime.values)


@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated.")
@pytest.mark.filterwarnings(
    "ignore:Changing the sparsity structure of a csr_matrix is expensive."
)
def test_run_palantir_int_generator_equivalence(mock_data):
    """``run_palantir(seed=42)`` == ``run_palantir(seed=default_rng(42))``."""
    r_int = run_palantir(mock_data, "cell_0", seed=42)
    r_gen = run_palantir(
        mock_data, "cell_0", seed=np.random.default_rng(42)
    )
    np.testing.assert_array_equal(
        r_int.pseudotime.values, r_gen.pseudotime.values
    )
