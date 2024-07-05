from __future__ import annotations

import numpy as np
import pytest
import torch

from qadence_embeddings.callable import ConcretizedCallable


@pytest.mark.parametrize(
    "fn", ["sin", "cos", "log", "tanh", "tan", "acos", "sin", "sqrt", "square"]
)
def test_univariate(fn: str) -> None:
    results = []
    x = np.random.uniform(0, 1)
    for engine_name in ["jax", "torch", "numpy"]:
        native_call = ConcretizedCallable(fn, ["x"], {}, engine_name)
        native_result = native_call(
            {"x": (torch.tensor(x) if engine_name == "torch" else x)}
        )
        results.append(native_result.item())
    assert np.allclose(results[0], results[1]) and np.allclose(results[0], results[2])


@pytest.mark.parametrize("fn", ["mul", "add", "div", "sub"])
def test_multivariate(fn: str) -> None:
    results = []
    x = np.random.randn(1)
    y = np.random.randn(1)
    for engine_name in ["jax", "torch", "numpy"]:
        native_call = ConcretizedCallable(fn, ["x", "y"], {}, engine_name)
        native_result = native_call(
            {
                "x": torch.tensor(x) if engine_name == "torch" else x,
                "y": torch.tensor(y) if engine_name == "torch" else y,
            }
        )
        results.append(native_result.item())
    assert np.allclose(results[0], results[1]) and np.allclose(results[0], results[2])
