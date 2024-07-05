from __future__ import annotations

from importlib import import_module
from logging import getLogger
from typing import Any, Optional

from numpy.typing import ArrayLike, DTypeLike

from .callable import ConcretizedCallable

logger = getLogger(__name__)


def init_param(
    engine_name: str, trainable: bool = True, device: str = "cpu"
) -> ArrayLike:
    engine = import_module(engine_name)
    if engine_name == "jax":
        return engine.random.uniform(engine.random.PRNGKey(42), shape=(1,))
    elif engine_name == "torch":
        return engine.rand(1, requires_grad=trainable, device=device)
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
        vparam_names: list[str] = [],
        fparam_names: list[str] = [],
        tparam_name: Optional[str] = None,
        var_to_call: dict[str, ConcretizedCallable] = dict(),
        engine_name: str = "torch",
        device: str = "cpu",
    ) -> None:

        self.vparams = {
            vp: init_param(engine_name, trainable=True, device=device)
            for vp in vparam_names
        }
        self.fparam_names: list[str] = fparam_names
        self.tparam_name = tparam_name
        self.var_to_call: dict[str, ConcretizedCallable] = var_to_call
        self._dtype: DTypeLike = None
        self.time_dependent_vars: list[str] = []

    @property
    def root_param_names(self) -> list[str]:
        return list(self.vparams.keys()) + self.fparam_names

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
            if self.tparam_name in engine_callable.abstract_args:
                self.time_dependent_vars.append(intermediate_or_leaf_var)
                # we remember which parameters depend on time
            inputs[intermediate_or_leaf_var] = engine_callable(inputs)
        return inputs

    def reembed_tparam(
        self,
        embedded_params: dict[str, ArrayLike],
        tparam_value: ArrayLike,
    ) -> dict[str, ArrayLike]:
        """Receive already embedded params containing intermediate and leaf parameters
        and recalculate the those which are dependent on the time parameter using the new value
        `tparam_value`.
        """
        assert self.tparam_name is not None
        embedded_params[self.tparam_name] = tparam_value
        for time_dependent_param in self.time_dependent_vars:
            embedded_params[time_dependent_param] = self.var_to_call[
                time_dependent_param
            ](embedded_params)
        return embedded_params

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
