from __future__ import annotations

import numpy as np
import torch

from qadence_embeddings.callable import ConcretizedCallable
from qadence_embeddings.embedding import Embedding


def test_embedding() -> None:
    x = np.random.uniform(0, 1)
    theta = np.random.uniform(0, 1)
    results = []
    for engine_name in ["jax", "torch", "numpy"]:
        v_params = ["theta"]
        f_params = ["x"]
        leaf0, native_call0 = "%0", ConcretizedCallable(
            "mul", ["x", "theta"], {}, engine_name
        )
        embedding = Embedding(
            v_params,
            f_params,
            None,
            var_to_call={leaf0: native_call0},
            engine_name=engine_name,
        )
        inputs = {
            "x": (torch.tensor(x) if engine_name == "torch" else x),
            "theta": (torch.tensor(theta) if engine_name == "torch" else theta),
        }
        eval_0 = embedding.evaluate_param("%0", inputs)
        results.append(eval_0.item())
    assert np.allclose(results[0], results[1]) and np.allclose(results[0], results[2])


def test_reembedding() -> None:
    x = np.random.uniform(0, 1)
    theta = np.random.uniform(0, 1)
    x_rembed = np.random.uniform(0, 1)
    results = []
    reembedded_results = []
    for engine_name in ["jax", "torch", "numpy"]:
        v_params = ["theta"]
        f_params = ["x"]
        leaf0, native_call0 = "%0", ConcretizedCallable(
            "mul", ["x", "theta"], {}, engine_name
        )
        embedding = Embedding(
            v_params,
            f_params,
            None,
            var_to_call={leaf0: native_call0},
            engine_name=engine_name,
        )
        inputs = {
            "x": (torch.tensor(x) if engine_name == "torch" else x),
            "theta": (torch.tensor(theta) if engine_name == "torch" else theta),
        }
        all_params = embedding.embed_all(
            inputs, include_root_vars=True, store_inputs=True
        )
        reembedded_params = embedding.reembed_all(
            {"x": (torch.tensor(x_rembed) if engine_name == "torch" else x_rembed)}
        )
        results.append(all_params["%0"].item())
        reembedded_results.append(reembedded_params["%0"].item())
    assert np.allclose(results[0], results[1]) and np.allclose(results[0], results[2])
    assert np.allclose(reembedded_results[0], reembedded_results[1]) and np.allclose(
        reembedded_results[0], reembedded_results[2]
    )
