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
        eval_0 = embedding.var_to_call["%0"](inputs)
        results.append(eval_0.item())
    assert np.allclose(results[0], results[1]) and np.allclose(results[0], results[2])


def test_reembedding() -> None:
    x = np.random.uniform(0, 1)
    theta = np.random.uniform(0, 1)
    t = np.random.uniform(0, 1)
    t_reembed = np.random.uniform(0, 1)
    results = []
    reembedded_results = []
    for engine_name in ["jax", "torch", "numpy"]:
        v_params = ["theta"]
        f_params = ["x"]
        tparam = "t"
        leaf0, native_call0 = "%0", ConcretizedCallable(
            "mul", ["x", "theta"], {}, engine_name
        )
        leaf1, native_call1 = "%1", ConcretizedCallable(
            "mul", ["t", "%0"], {}, engine_name
        )

        leaf2, native_call2 = "%2", ConcretizedCallable("sin", ["%1"], {}, engine_name)
        embedding = Embedding(
            v_params,
            f_params,
            tparam,
            var_to_call={leaf0: native_call0, leaf1: native_call1, leaf2: native_call2},
            engine_name=engine_name,
        )
        inputs = {
            "x": (torch.tensor(x) if engine_name == "torch" else x),
            "theta": (torch.tensor(theta) if engine_name == "torch" else theta),
            "t": (torch.tensor(t) if engine_name == "torch" else t),
        }
        all_params = embedding.embed_all(inputs)
        new_tparam_val = (
            torch.tensor(t_reembed) if engine_name == "torch" else t_reembed
        )
        reembedded_params = embedding.reembed_time(all_params, new_tparam_val)
        results.append(all_params["%2"].item())
        reembedded_results.append(reembedded_params["%2"].item())
    assert all([p in ["%1", "%2"] for p in embedding.time_dependent_vars])
    assert "%0" not in embedding.time_dependent_vars
    assert np.allclose(results[0], results[1]) and np.allclose(results[0], results[2])
    assert np.allclose(reembedded_results[0], reembedded_results[1]) and np.allclose(
        reembedded_results[0], reembedded_results[2]
    )
