"""Feature normalization with leakage-safe statistics.

Despite the spec filename (``normalization.py``) and historical
"rolling-normalization" language, the implementation here is explicitly
**not** rolling / online. The only leakage-safe choice for offline RL on
financial time series is to fit statistics once on the training window and
reuse them unchanged for validation and test. Rolling normalization at test
time would implicitly leak information about future return distributions
into the scaled features, inflating backtest metrics in a way that does not
hold out of sample.

Design points
-------------
* **float32 everywhere.** Downstream torch networks expect float32; casting
  at the normalizer boundary saves a copy later and catches dtype drift.
* **``eps`` floor on std.** Zero-variance columns (e.g., a feature that is
  constant on the train window) would cause divide-by-zero. We clip the
  learned std to at least ``eps`` so constant features map cleanly to 0.
* **Persistence.** ``save`` / ``load`` use plain JSON so normalizer state
  is human-readable and safe to commit alongside experiment artifacts.
* **Refuses to transform before fitting**, and refuses to transform inputs
  whose feature dimensionality disagrees with the fit data. Both are common
  sources of silent bugs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["CausalZScoreNormalizer"]


class CausalZScoreNormalizer:
    """Per-feature z-score normalizer fit on the training split only.

    Parameters
    ----------
    eps:
        Lower bound on per-feature standard deviation. Protects against
        division-by-zero on constant columns. Default: ``1e-8``.

    Examples
    --------
    >>> X_train = np.random.randn(1000, 4).astype(np.float32)
    >>> norm = CausalZScoreNormalizer().fit(X_train)
    >>> X_test = np.random.randn(100, 4).astype(np.float32)
    >>> Z_test = norm.transform(X_test)  # uses train mean/std
    >>> norm.save("results/run42/normalizer.json")
    """

    def __init__(self, eps: float = 1e-8) -> None:
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps: float = float(eps)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Core fit / transform
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> CausalZScoreNormalizer:
        """Compute per-column mean and std over the training window.

        Parameters
        ----------
        X:
            Training feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        Self, to allow chaining ``CausalZScoreNormalizer().fit(X)``.
        """
        X = self._check_array(X, label="X", require_2d=True)
        self.mean_ = X.mean(axis=0).astype(np.float32)
        # Population std (ddof=0) is the right choice here: we want the
        # actual observed scale, not an unbiased estimator.
        raw_std = X.std(axis=0, ddof=0).astype(np.float32)
        self.std_ = np.maximum(raw_std, np.float32(self.eps))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply ``(X - mean) / std`` using the *fit-time* statistics.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of columns than the fit data.
        """
        if not self.fitted:
            raise RuntimeError(
                "CausalZScoreNormalizer must be fit before calling transform(). "
                "Call .fit(X_train) first."
            )
        X = self._check_array(X, label="X", require_2d=True)
        assert self.mean_ is not None and self.std_ is not None  # for type narrowing
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"transform() got feature dim {X.shape[1]}, "
                f"but normalizer was fit with feature dim {self.mean_.shape[0]}."
            )
        return ((X.astype(np.float32) - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit on ``X`` then transform ``X``."""
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        """True iff ``fit`` has been called successfully."""
        return self.mean_ is not None and self.std_ is not None

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the normalizer state."""
        if not self.fitted:
            raise RuntimeError("Cannot serialize an unfitted normalizer.")
        assert self.mean_ is not None and self.std_ is not None
        return {
            "eps": self.eps,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "version": 1,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """In-place restore from a dict produced by :meth:`state_dict`."""
        required = {"eps", "mean", "std"}
        missing = required - state.keys()
        if missing:
            raise KeyError(f"state_dict missing required keys: {sorted(missing)}")
        self.eps = float(state["eps"])
        self.mean_ = np.asarray(state["mean"], dtype=np.float32)
        self.std_ = np.asarray(state["std"], dtype=np.float32)
        if self.mean_.shape != self.std_.shape:
            raise ValueError(
                f"mean shape {self.mean_.shape} and std shape "
                f"{self.std_.shape} must match."
            )

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> CausalZScoreNormalizer:
        """Construct a new normalizer from a state dict."""
        norm = cls(eps=float(state.get("eps", 1e-8)))
        norm.load_state_dict(state)
        return norm

    def save(self, path: str | Path) -> None:
        """Persist state to a JSON file; auto-creates parent directories."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CausalZScoreNormalizer:
        """Load a normalizer from a JSON file written by :meth:`save`."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            return cls.from_state_dict(json.load(f))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_array(X: np.ndarray, *, label: str, require_2d: bool) -> np.ndarray:
        """Coerce input to a contiguous float ndarray with dimension checks."""
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if require_2d and X.ndim != 2:
            raise ValueError(
                f"{label} must be 2D (n_samples, n_features); got shape {X.shape}."
            )
        if not np.issubdtype(X.dtype, np.floating):
            X = X.astype(np.float64)
        return np.ascontiguousarray(X)
