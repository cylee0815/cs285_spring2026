"""Tests for ``features.normalization.CausalZScoreNormalizer``.

These tests are written **before** the implementation. The normalizer is the
primary defense against train/test leakage in feature scaling — every
invariant below exists because violating it would leak future distribution
information into the training objective.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from features.normalization import CausalZScoreNormalizer


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded numpy RNG for reproducible random data."""
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Fit / transform basics
# ---------------------------------------------------------------------------


def test_fit_transform_produces_zero_mean_unit_std(rng: np.random.Generator) -> None:
    """fit_transform on the fitting data should yield mean≈0, std≈1 per column."""
    X = rng.normal(loc=[3.0, -2.0, 10.0], scale=[0.5, 2.0, 5.0], size=(1000, 3))
    norm = CausalZScoreNormalizer()
    Z = norm.fit_transform(X)

    assert Z.shape == X.shape
    assert Z.dtype == np.float32  # normalizer outputs float32 for torch interop
    np.testing.assert_allclose(Z.mean(axis=0), np.zeros(3), atol=1e-5)
    np.testing.assert_allclose(Z.std(axis=0, ddof=0), np.ones(3), atol=1e-5)


def test_transform_uses_fit_statistics_not_transform_statistics(rng: np.random.Generator) -> None:
    """Transforming a *different* array must reuse the fit mean/std, not refit."""
    X_train = rng.normal(loc=0.0, scale=1.0, size=(500, 4))
    X_test = rng.normal(loc=5.0, scale=3.0, size=(200, 4))

    norm = CausalZScoreNormalizer().fit(X_train)
    Z_test = norm.transform(X_test)

    # If the normalizer (wrongly) refit on X_test, Z_test would have mean≈0.
    # Because it uses train stats, mean(Z_test) ≈ (5 - 0) / 1 = 5 and
    # std(Z_test) ≈ 3 / 1 = 3.
    assert np.all(np.abs(Z_test.mean(axis=0) - 5.0) < 0.5)
    assert np.all(np.abs(Z_test.std(axis=0) - 3.0) < 0.5)


def test_fit_returns_self_for_chaining(rng: np.random.Generator) -> None:
    X = rng.normal(size=(100, 2))
    norm = CausalZScoreNormalizer()
    result = norm.fit(X)
    assert result is norm


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


def test_zero_variance_column_does_not_produce_nan() -> None:
    """Constant feature columns must not blow up the normalizer."""
    X = np.tile(np.array([1.0, 2.0, 3.0]), (50, 1))  # every row identical → zero std
    norm = CausalZScoreNormalizer(eps=1e-8)
    Z = norm.fit_transform(X)
    assert not np.isnan(Z).any()
    assert not np.isinf(Z).any()
    # Because the variance is 0 and we clip std to eps, the result is 0 for
    # every entry (x - mean == 0, then / eps == 0).
    np.testing.assert_allclose(Z, np.zeros_like(Z), atol=1e-5)


def test_eps_floor_prevents_explosion() -> None:
    """A tiny-variance column should be scaled by eps, not by its actual std."""
    X = np.zeros((100, 1))
    X[::2, 0] = 1e-12
    X[1::2, 0] = -1e-12
    norm = CausalZScoreNormalizer(eps=1e-6)
    norm.fit(X)
    # Use np.float32(eps) for the comparison: the normalizer stores std_ as
    # float32 and clips it to float32(eps), which in float32 arithmetic may
    # differ from the float64 literal 1e-6 by ~5e-15.
    assert norm.std_[0] >= np.float32(norm.eps)
    # And the transformed output must be finite.
    Z = norm.transform(X)
    assert np.isfinite(Z).all()


def test_transform_before_fit_raises() -> None:
    X = np.zeros((10, 3))
    norm = CausalZScoreNormalizer()
    with pytest.raises(RuntimeError, match="fit"):
        norm.transform(X)


def test_fit_rejects_1d_array() -> None:
    """Guard against common mistake of passing a flat vector."""
    norm = CausalZScoreNormalizer()
    with pytest.raises(ValueError, match="2D"):
        norm.fit(np.array([1.0, 2.0, 3.0]))


def test_transform_rejects_wrong_feature_dim(rng: np.random.Generator) -> None:
    """Transform must refuse inputs whose column count disagrees with fit."""
    X_train = rng.normal(size=(100, 4))
    X_wrong = rng.normal(size=(100, 5))
    norm = CausalZScoreNormalizer().fit(X_train)
    with pytest.raises(ValueError, match="feature dim"):
        norm.transform(X_wrong)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_state_dict_roundtrip(rng: np.random.Generator) -> None:
    X = rng.normal(loc=1.5, scale=2.5, size=(300, 3))
    norm = CausalZScoreNormalizer().fit(X)
    state = norm.state_dict()
    restored = CausalZScoreNormalizer.from_state_dict(state)
    np.testing.assert_allclose(norm.mean_, restored.mean_, atol=1e-7)
    np.testing.assert_allclose(norm.std_, restored.std_, atol=1e-7)

    # Restored normalizer should transform identically to the original.
    X_test = rng.normal(size=(50, 3))
    np.testing.assert_allclose(
        norm.transform(X_test), restored.transform(X_test), atol=1e-7
    )


def test_save_load_roundtrip(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    X = rng.normal(loc=0.5, scale=1.5, size=(200, 2))
    norm = CausalZScoreNormalizer().fit(X)
    path = tmp_path / "norm.json"
    norm.save(path)
    assert path.exists()

    loaded = CausalZScoreNormalizer.load(path)
    X_test = rng.normal(size=(20, 2))
    np.testing.assert_allclose(
        norm.transform(X_test), loaded.transform(X_test), atol=1e-7
    )


def test_load_creates_parent_directories(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    """save() should auto-create missing parent directories for ergonomics."""
    X = rng.normal(size=(50, 2))
    norm = CausalZScoreNormalizer().fit(X)
    path = tmp_path / "nested" / "dirs" / "norm.json"
    norm.save(path)
    assert path.exists()


def test_fitted_property(rng: np.random.Generator) -> None:
    norm = CausalZScoreNormalizer()
    assert norm.fitted is False
    norm.fit(rng.normal(size=(10, 2)))
    assert norm.fitted is True
