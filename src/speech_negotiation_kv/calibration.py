from __future__ import annotations

import numpy as np


def fit_ridge_opponent_code(C: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError("C must be a 2D design matrix")
    if y.ndim != 1 or len(y) != len(C):
        raise ValueError("y must be a 1D vector aligned with C")
    eye = np.eye(C.shape[1], dtype=np.float64)
    return np.linalg.solve(C.T @ C + ridge * eye, C.T @ y)
