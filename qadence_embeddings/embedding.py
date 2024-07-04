from __future__ import annotations

from importlib import import_module
from logging import getLogger
from typing import Any, Optional

from numpy.typing import ArrayLike, DTypeLike

from .callable import ConcretizedCallable

logger = getLogger(__name__)


def init_param(engine_name: str, trainable: bool = True) -> ArrayLike:
    engine = import_module(engine_name)
    if engine_name == "jax":
        return engine.random.uniform(engine.random.PRNGKey(42), shape=(1,))
    elif engine_name == "torch":
        return engine.rand(1, requires_grad=trainable)
    elif engine_name == "numpy":
        return engine.random.uniform(0, 1)


class Embedding:
    """
    A generic module class to hold and handle the parameters and expressions
    functions coming from the `Model`. It may contain the list of user input
    parameters, as well as the trainable variational parameters and the
    evaluated functions from the data types being used, i.e. torch, numpy, etc.
    """

    def __init__(
        self,
        vparam_names: list[str],
        fparam_names: list[str],
        tparam_names: Optional[list[str]],
        var_to_call: dict[str, ConcretizedCallable],
        engine_name: str = "torch",
    ) -> None:
        self.vparams = {
            vp: init_param(engine_name, trainable=True) for vp in vparam_names
        }
        self.fparams: dict[str, Optional[ArrayLike]] = {fp: None for fp in fparam_names}
        self.tparams: dict[str, Optional[ArrayLike]] = (
            None
            if tparam_names is None
            else {fp: None for fp in tparam_names}  #  type: ignore[assignment]
        )
        self.var_to_call: dict[str, ConcretizedCallable] = var_to_call
        self._dtype: DTypeLike = None

    @property
    def root_param_names(self) -> list[str]:
        return list(self.vparams.keys()) + list(self.fparams.keys())

    def embed_all(
        self,
        inputs: dict[str, ArrayLike],
    ) -> dict[str, ArrayLike]:
        """The standard embedding of all intermediate and leaf parameters.
        Include the root_params, i.e., the vparams and fparams original values
        to be reused in computations.
        """
        for intermediate_or_leaf_var, engine_callable in self.var_to_call.items():
            # We mutate the original inputs dict and include intermediates and leaves.
            inputs[intermediate_or_leaf_var] = engine_callable(inputs)
        return inputs

    def reembed_all(
        self,
        embedded_params: dict[str, ArrayLike],
        new_root_params: dict[str, ArrayLike],
    ) -> dict[str, ArrayLike]:
        """Receive already embedded params containing intermediate and leaf parameters
        and remove them from the `embedded_params` dict to reconstruct the user input, and finally
        recalculate the embedding using values for parameters in passes in `new_root_params`.
        """
        # We filter out intermediates and leaves and leave only the original vparams and fparams +
        # the `inputs` dict which contains new <name:parameter value> pairs
        inputs = {
            p: v for p, v in embedded_params.items() if p in self.root_param_names
        }
        return self.embed_all({**self.vparams, **inputs, **new_root_params})

    def __call__(self, inputs: dict[str, ArrayLike]) -> dict[str, ArrayLike]:
        """Functional version of legacy embedding: Return a new dictionary\
        with all embedded parameters."""
        return self.embed_all(inputs)

    @property
    def dtype(self) -> DTypeLike:
        return self._dtype

    def to(self, args: Any, kwargs: Any) -> None:
        # TODO move to device and dtype
        pass
