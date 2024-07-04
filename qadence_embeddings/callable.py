from __future__ import annotations

from importlib import import_module
from logging import getLogger
from typing import Tuple

from numpy.typing import ArrayLike

logger = getLogger(__name__)

ARRAYLIKE_FN_MAP = {
    "torch": ("torch", "tensor"),
    "jax": ("jax.numpy", "array"),
    "numpy": ("numpy", "array"),
}


DEFAULT_JAX_MAPPING = {
    "mul": ("jax.numpy", "multiply"),
    "sub": ("jax.numpy", "subtract"),
    "div": ("jax.numpy", "divide"),
}
DEFAULT_TORCH_MAPPING: dict = {}
DEFAULT_NUMPY_MAPPING = {
    "mul": ("numpy", "multiply"),
    "sub": ("numpy", "subtract"),
    "div": ("numpy", "divide"),
}

DEFAULT_INSTRUCTION_MAPPING = {
    "torch": DEFAULT_TORCH_MAPPING,
    "jax": DEFAULT_JAX_MAPPING,
    "numpy": DEFAULT_NUMPY_MAPPING,
}


class ConcretizedCallable:

    def __init__(
        self,
        call_name: str,
        abstract_args: list[str | float | int],
        instruction_mapping: dict[str, Tuple[str, str]] = dict(),
        engine_name: str = "torch",
    ) -> None:
        instruction_mapping = {
            **instruction_mapping,
            **DEFAULT_INSTRUCTION_MAPPING[engine_name],
        }
        self.call_name = call_name
        self.abstract_args = abstract_args
        self.engine_name = engine_name
        self.engine_call = None
        engine_call = None
        engine = None
        try:
            engine_name, fn_name = ARRAYLIKE_FN_MAP[engine_name]
            engine = import_module(engine_name)
            self.arraylike_fn = getattr(engine, fn_name)
        except (ModuleNotFoundError, ImportError) as e:
            logger.error(f"Unable to import {engine_call} due to {e}.")

        try:
            try:
                self.engine_call = getattr(engine, call_name)
            except AttributeError:
                pass
            if self.engine_call is None:
                mod, fn = instruction_mapping[call_name]
                self.engine_call = getattr(import_module(mod), fn)
        except (ImportError, KeyError) as e:
            logger.error(
                f"Requested function {call_name} can not be imported from {engine_name} and is\
                        not in instruction_mapping {instruction_mapping} due to {e}."
            )

    """Convert a generic abstract function call and
    a list of symbolic or constant parameters
    into a concretized Callable in a particular engine.
    which can be evaluated using
    a inputs dict.

    Arguments:
        call_name: The name of the function
        abstract_args: A list of strings (in the case of parameters) and numeric constants
                        denoting the arguments for `call_name`
        instruction_mapping: A dict mapping from an abstract call_name to its name in an engine.
        engine_name: The engine to use to create the callable.

    Example:
    ```
    In [11]: call = ConcretizedCallable('sin', ['x'], engine_name='numpy')
    In [12]: call({'x': 0.5})
    Out[12]: 0.479425538604203

    In [13]: call = ConcretizedCallable('sin', ['x'], engine_name='torch')
    In [14]: call({'x': torch.rand(1)})
    Out[14]: tensor([0.5531])

    In [15]: call = ConcretizedCallable('sin', ['x'], engine_name='jax')
    In [16]: call({'x': 0.5})
    Out[16]: Array(0.47942555, dtype=float32, weak_type=True)
    ```
    """

    def evaluate(self, inputs: dict[str, ArrayLike] = dict()) -> ArrayLike:
        arraylike_args = []
        for symbol_or_numeric in self.abstract_args:
            if isinstance(symbol_or_numeric, (float, int)):
                arraylike_args.append(self.arraylike_fn(symbol_or_numeric))
            elif isinstance(symbol_or_numeric, str):
                arraylike_args.append(inputs[symbol_or_numeric])
        return self.engine_call(*arraylike_args)  # type: ignore[misc]

    def __call__(self, inputs: dict[str, ArrayLike] = dict()) -> ArrayLike:
        return self.evaluate(inputs)
