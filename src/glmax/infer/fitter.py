from abc import abstractmethod

import equinox as eqx

from jaxtyping import ArrayLike

from .result import GLMState


class AbstractFitter(eqx.Module, strict=True):
    @abstractmethod
    def __call__(
        self,
        model: object,
        X: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        *,
        init: ArrayLike | None = None,
        options: dict[str, object] | None = None,
    ) -> GLMState:
        pass


class DefaultFitter(AbstractFitter, strict=True):
    def __call__(
        self,
        model: object,
        X: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike,
        *,
        init: ArrayLike | None = None,
        options: dict[str, object] | None = None,
    ) -> GLMState:
        fit_options = {} if options is None else dict(options)
        from ..fit import _run_default_pipeline

        covariance = fit_options.pop("se_estimator", None)
        test_hook = fit_options.pop("test_hook", None)
        return _run_default_pipeline(
            model,
            X,
            y,
            offset,
            init=init,
            covariance=covariance,
            tests=test_hook,
            options=fit_options,
        )
