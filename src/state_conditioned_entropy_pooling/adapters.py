from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .probabilities import normalize_probabilities


@dataclass
class ScenarioSupport:
    """Minimal interface between a physical scenario engine and the conditioning layer."""
    instrument_returns: pd.DataFrame
    scenario_features: pd.DataFrame
    prior: np.ndarray | None = None

    def __post_init__(self):
        if len(self.instrument_returns) != len(self.scenario_features):
            raise ValueError("instrument_returns and scenario_features must have the same rows")
        if self.prior is None:
            self.prior = np.full(len(self.instrument_returns), 1.0 / len(self.instrument_returns))
        else:
            self.prior = normalize_probabilities(self.prior)
        if len(self.prior) != len(self.instrument_returns):
            raise ValueError("prior length does not match scenario support")

    @classmethod
    def from_arrays(cls, instrument_returns, scenario_features, *, asset_names=None, feature_names=None, prior=None):
        R = np.asarray(instrument_returns, dtype=float)
        F = np.asarray(scenario_features, dtype=float)
        asset_names = asset_names or [f"asset_{i}" for i in range(R.shape[1])]
        feature_names = feature_names or [f"feature_{i}" for i in range(F.shape[1])]
        return cls(pd.DataFrame(R, columns=asset_names), pd.DataFrame(F, columns=feature_names), prior=prior)
