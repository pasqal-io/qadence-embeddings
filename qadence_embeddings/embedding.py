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
        self.fparams_assigned: bool = False

    def flush_fparams(self) -> None:
        """Flush all stored fparams and set them to None."""
        self.fparams = {key: None for key in self.fparams.keys()}
        self.fparams_assigned = False

    def assign_fparams(
        self, inputs: dict[str, ArrayLike | None], flush_current: bool = True
    ) -> None:
        """Mutate the `self.fparams` field to store inputs from the user."""
        if self.fparams_assigned:
            (
                self.flush_fparams()
                if flush_current
                else logger.error(
                    "Fparams are still assigned. Please flush them before re-embedding."
                )
            )
        if not inputs.keys() == self.fparams.keys():
            logger.error(
                f"Please provide all fparams, Expected {self.fparams.keys()},\
                  received {inputs.keys()}."
            )
        self.fparams = inputs
        self.fparams_assigned = True

    def evaluate_param(
        self, param_name: str, inputs: dict[str, ArrayLike]
    ) -> ArrayLike:
        """Returns the result of evaluation an expression in `var_to_call`."""
        return self.var_to_call[param_name](inputs)

    def embed_all(
        self,
        inputs: dict[str, ArrayLike],
        include_root_vars: bool = True,
        store_inputs: bool = False,
    ) -> dict[str, ArrayLike]:
        """The standard embedding of all intermediate and leaf parameters.
        Include the root_params, i.e., the vparams and fparams original values
        to be reused in computations.
        """
        if not include_root_vars:
            logger.error(
                "Warning: Original parameters are not included, only intermediates and leaves."
            )
        if store_inputs:
            self.assign_fparams(inputs)
        for intermediate_or_leaf_var, engine_callable in self.var_to_call.items():
            # We mutate the original inputs dict and include intermediates and leaves.
            inputs[intermediate_or_leaf_var] = engine_callable(inputs)
        return inputs

    def reembed_all(self, inputs: dict[str, ArrayLike]) -> dict[str, ArrayLike]:
        assert (
            self.fparams_assigned
        ), "To reembed, please store original fparam values by setting\
        `include_root_vars = True` when calling `embed_all`"

        # We filter out intermediates and leaves and leave only the original vparams and fparams +
        # the `inputs` dict which contains new <name:parameter value> pairs
        inputs = {
            p: v
            for p, v in inputs.items()
            if p in self.vparams.keys() or p in self.fparams.keys()
        }
        return self.embed_all({**self.vparams, **self.fparams, **inputs})

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
