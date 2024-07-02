from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class ParameterType:
    pass


class DType:
    pass


class Embedding(ABC):
    """
    A generic module class to hold and handle the parameters and expressions
    functions coming from the `Model`. It may contain the list of user input
    parameters, as well as the trainable variational parameters and the
    evaluated functions from the data types being used, i.e. torch, numpy, etc.
    """

    vparams: dict[str, ParameterType]
    fparams: dict[str, Optional[ParameterType]]
    mapped_vars: dict[str, Callable]
    _dtype: DType

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Callable]:
        raise NotImplementedError()

    @abstractmethod
    def name_mapping(self) -> dict:
        raise NotImplementedError()

    @property
    def dtype(self) -> DType:
        return self._dtype
