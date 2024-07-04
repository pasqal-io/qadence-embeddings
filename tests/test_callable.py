from __future__ import annotations

import numpy as np
import torch

from qadence_embeddings.callable import ConcretizedCallable


def test_sin() -> None:
    results = []
    x = np.random.randn(1)
    for engine_name in ["jax", "torch", "numpy"]:
        native_call = ConcretizedCallable("sin", ["x"], {}, engine_name)
        native_result = native_call(
            {"x": (torch.tensor(x) if engine_name == "torch" else x)}
        )
        results.append(native_result.item())
    assert np.allclose(results[0], results[1]) and np.allclose(results[0], results[2])


def test_log() -> None:
    results = []
    x = np.random.uniform(0, 5)
    for engine_name in ["jax", "torch", "numpy"]:
        native_call = ConcretizedCallable("log", ["x"], {}, engine_name)
        native_result = native_call(
            {"x": (torch.tensor(x) if engine_name == "torch" else x)}
        )
        results.append(native_result.item())

    assert np.allclose(results[0], results[1]) and np.allclose(results[0], results[2])


def test_mul() -> None:
    results = []
    x = np.random.randn(1)
    y = np.random.randn(1)
    for engine_name in ["jax", "torch", "numpy"]:
        native_call = ConcretizedCallable("mul", ["x", "y"], {}, engine_name)
        native_result = native_call(
            {
                "x": torch.tensor(x) if engine_name == "torch" else x,
                "y": torch.tensor(y) if engine_name == "torch" else y,
            }
        )
        results.append(native_result.item())
    assert np.allclose(results[0], results[1]) and np.allclose(results[0], results[2])


def test_add() -> None:
    results = []
    x = np.random.randn(1)
    y = np.random.randn(1)
    for engine_name in ["jax", "torch", "numpy"]:
        native_call = ConcretizedCallable("add", ["x", "y"], {}, engine_name)
        native_result = native_call(
            {
                "x": torch.tensor(x) if engine_name == "torch" else x,
                "y": torch.tensor(y) if engine_name == "torch" else y,
            }
        )
        results.append(native_result.item())
    assert np.allclose(results[0], results[1]) and np.allclose(results[0], results[2])
