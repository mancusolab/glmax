# glmax

## Purpose
`glmax` provides grammar-first generalized linear modeling in JAX. Keep the public surface centered on explicit nouns (`Params`, `FitResult`, `FittedGLM`, `InferenceResult`, `AbstractDiagnostic`, `GofStats`, `InfluenceStats`) and verbs (`fit`, `predict`, `infer`, `check`) rather than wrapper-heavy or module-internal APIs.

## Contracts
- **Exposes**:
  - Package-root API from `src/glmax/__init__.py`: `Params`, `AbstractFitter`, `FitResult`, `FittedGLM`, `IRLSFitter`, `NewtonFitter`, `InferenceResult`, `AbstractDiagnostic`, `PearsonResidual`, `DevianceResidual`, `QuantileResidual`, `GoodnessOfFit`, `GofStats`, `Influence`, `InfluenceStats`, `AbstractTest`, `WaldTest`, `ScoreTest`, `AbstractStdErrEstimator`, `FisherInfoError`, `HuberError`, `fit`, `predict`, `infer`, `check`, `AbstractLink`, `IdentityLink`, `LogLink`, `LogitLink`, `InverseLink`, `PowerLink`, `ProbitLink`, `CLogLogLink`, `LogLogLink`, `SqrtLink`, `CauchitLink`, `NBLink`, `Gaussian`, `Gamma`, `Poisson`, `Binomial`, `NegativeBinomial`, `ExponentialDispersionFamily`.
  - Family and link implementations from `src/glmax/family/__init__.py`.
  - User-facing grammar docs in `README.md`, `docs/index.md`, `docs/api/families-and-links.md`, `docs/api/fit/index.md`, `docs/api/fit/strategies.md`, `docs/api/predict.md`, `docs/api/infer/index.md`, `docs/api/infer/strategies.md`, and `docs/api/check.md`.
- **Guarantees**:
  - Canonical user workflow is `fit -> predict -> infer -> check`.
  - `glmax.fit(family, X, y, *, offset=None, weights=None, init=None, fitter=IRLSFitter())` is the curated public fit contract, is `@eqx.filter_jit`-wrapped, and returns `FittedGLM`.
  - `glmax.predict(family, params, X, *, offset=None)` is also `@eqx.filter_jit`-wrapped and returns predicted means $\hat{\mu}$.
  - `infer(fitted, inferrer=WaldTest(), stderr=FisherInfoError())` and `check(fitted, diagnostic=...)` operate on the fitted noun without refitting.
  - `FittedGLM` binds `family` and `result` (`FitResult`). Access fitted artifacts via `fitted.family`, `fitted.params`, `fitted.beta`, `fitted.eta`, `fitted.mu`, etc.
  - `Gamma` is a supported family (exported from `glmax.family`). Dispersion estimation for Gamma is deferred.
  - Per-sample `weights` are accepted by the `fit` signature but raise `ValueError` until implemented.
- **Expects**:
  - `X` passed to `fit` is rank-2 with shape `(n, p)`; `y` is rank-1 with shape `(n,)`. Both must contain only finite values.
  - Optional `offset` and `weights` broadcast over the sample axis when present.
  - `Params.beta` is an inexact rank-1 vector of length `p`; `Params.disp` is an inexact scalar; `Params.aux` is either `None` or an inexact family-specific scalar.
  - `FitResult` is the fitter contract and carries `params`, `X`, `y`, `eta`, `mu`, `glm_wt`, `converged`, `num_iters`, `objective`, `objective_delta`, and `score_residual`.
  - `FittedGLM` is the public fitted noun and binds `family` plus `result`, forwarding common fit artifacts for ergonomics.
  - `InferenceResult` carries `params`, `se`, `stat`, and `p`; those summaries are produced by `infer(fitted, ...)`, not `fit(...)`.
  - `check(fitted, diagnostic=...)` is `@eqx.filter_jit`-wrapped and returns a single typed diagnostic result `T` for the supplied `AbstractDiagnostic[T]`. `check(fitted)` uses the function's default diagnostic without refitting.
  - `AbstractFitter` subclasses must declare `solver`, `step_size`, `tol`, and `max_iter` as concrete fields. `IRLSFitter` defaults to `solver=lx.Cholesky()`, `step_size=1.0`, `tol=1e-3`, `max_iter=1000`. `NewtonFitter` adds `armijo_c=0.1` and `armijo_factor=0.5`; its `step_size` is the initial Armijo trial step (default `1.0`) and its default `tol=1e-6`, `max_iter=200`.
  - Negative Binomial stores its auxiliary `alpha` in `Params.aux`; canonical `Params.disp` remains the GLM dispersion slot and is `1.0` for Negative Binomial fits.

## Dependencies
- **Uses**: `jax`, `jaxlib`, `equinox`, `jaxtyping`, `lineax`, `optimistix`.
- **Boundary**:
  - `src/glmax/_fit/types.py` owns `Params`, `FitResult`, `FittedGLM`, and `AbstractFitter` (abstract base with `solver`, `step_size`, `tol`, `max_iter` as `AbstractVar` fields).
  - `src/glmax/_fit/fit.py` owns the public `fit` and `predict` verbs (`@eqx.filter_jit`-wrapped). Input validation (shape, finiteness) lives here.
  - `src/glmax/_fit/irls.py` owns `IRLSFitter` (the default fitter) and the `_irls` kernel. `_irls` takes raw JAX arrays and calls family methods directly.
  - `src/glmax/_fit/__init__.py` re-exports all fit internals.
  - `src/glmax/diagnostics.py` owns `AbstractDiagnostic`, the built-in diagnostic strategies/results, and `check`.
  - `src/glmax/_infer/infer.py` owns `infer`.
  - `src/glmax/_infer/types.py` owns `InferenceResult`.
  - `src/glmax/_infer/hyptest.py` owns `AbstractTest`, `WaldTest`, `ScoreTest`.
  - `src/glmax/_infer/stderr.py` owns `AbstractStdErrEstimator`, `FisherInfoError`, `HuberError`.
  - `src/glmax/_infer/__init__.py` re-exports all infer internals.
  - Other modules under `src/glmax/_infer/` are internal numerics seams, not package-root API.

## Invariants
- Contract carrier split is deliberate: `FitResult` and `FittedGLM` are `equinox.Module` types with constructor-time validation; `Params` and `InferenceResult` are `NamedTuple` pytrees; diagnostic strategies/results are `equinox.Module` types and arrays returned through `check(...)`.
- Public exports stay centralized in `src/glmax/__init__.py`; if a user-facing noun or verb changes, update docs and contract tests in the same patch.
- Keep workflow examples on the grammar nouns and top-level verbs. Reserve `_fit` imports for fitter/solver docs and tests; do not present `_fit` or `_infer` module paths as the primary user workflow.
- `site/` is generated documentation output. Edit `docs/` and `mkdocs.yml`, not `site/`.

## Verification
- Do not run bare `pytest`.
- Use `pytest -p no:capture tests` for all pytest invocations.
- Keep `README.md`, `docs/index.md`, `docs/api/families-and-links.md`, `docs/api/fit/index.md`, `docs/api/fit/strategies.md`, `docs/api/predict.md`, `docs/api/infer/index.md`, `docs/api/infer/strategies.md`, and `docs/api/check.md` aligned with the package-root exports and advanced strategy surface.
- Contract changes should update the owning tests, especially `tests/package/test_api.py`, `tests/package/test_grammar.py`, and the relevant verb-specific suites under `tests/fit/`, `tests/infer/`, `tests/data/`, `tests/glm/`.
- Public numerics and solver docstrings use raw markdown section labels: `**Arguments:**`, `**Returns:**`, and `**Raises:**` or `**Failure Modes:**`.
