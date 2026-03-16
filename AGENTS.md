# glmax

## Purpose
`glmax` provides grammar-first generalized linear modeling in JAX. Keep the public surface centered on explicit nouns (`GLMData`, `Params`, `FitResult`, `FittedGLM`, `InferenceResult`, `Diagnostics`) and verbs (`specify`, `fit`, `predict`, `infer`, `check`) rather than wrapper-heavy or module-internal APIs.

## Contracts
- **Exposes**:
  - Package-root API from `src/glmax/__init__.py`: `GLMData`, `Params`, `GLM`, `AbstractFitter`, `FitResult`, `FittedGLM`, `InferenceResult`, `Diagnostics`, `AbstractTest`, `WaldTest`, `ScoreTest`, `AbstractStdErrEstimator`, `FisherInfoError`, `HuberError`, `specify`, `predict`, `fit`, `infer`, `check`.
  - Family and link implementations from `src/glmax/family/__init__.py`.
  - User-facing grammar docs in `README.md`, `docs/index.md`, and `docs/api/glm.md`.
- **Guarantees**:
  - Canonical user workflow is `specify -> fit -> predict -> infer -> check`.
  - `glmax.fit(model, data, init=None, *, fitter=IRLSFitter())` is the curated public fit contract, is `@eqx.filter_jit`-wrapped, and returns `FittedGLM`.
  - `glmax.predict(model, params, data)` is also `@eqx.filter_jit`-wrapped.
  - `infer(fitted, inferrer=WaldTest(), stderr=FisherInfoError())` and `check(fitted)` operate on the fitted noun without refitting.
  - `GLM` is a pure specification noun (`family` field only). It has no `.fit` method and no `solver` field. Use `glmax.fit(model, data)`. The solver lives on the fitter strategy.
  - `Gamma` is a supported family (exported from `glmax.family`). Dispersion estimation for Gamma is deferred.
- **Expects**:
  - `GLMData.X` is rank-2 with shape `(n, p)` and `GLMData.y` is rank-1 with shape `(n,)`.
  - Optional `offset` and `weights` inputs broadcast over the sample axis when present.
  - `Params.beta` is an inexact rank-1 vector of length `p`; `Params.disp` is an inexact scalar.
  - `FitResult` is the fitter contract and carries `params`, `X`, `y`, `eta`, `mu`, `glm_wt`, `converged`, `num_iters`, `objective`, `objective_delta`, and `score_residual`.
  - `FittedGLM` is the public fitted noun and binds `model` plus `result`, forwarding common fit artifacts for ergonomics.
  - `InferenceResult` carries `params`, `se`, `stat`, and `p`; those summaries are produced by `infer(fitted, ...)`, not `fit(...)`.
  - `Diagnostics` is reserved for `check(fitted)` model-fit assessment. The current `check()` seam is a placeholder contract and does not yet expose finalized model-diagnostic fields.
  - `AbstractFitter` subclasses must declare a `solver: AbstractLinearSolver` field. `IRLSFitter` defaults to `CholeskySolver()`.
  - `GLMData.weights` is not part of the supported public `fit` or `predict` contract unless docs and tests are updated in the same change.

## Dependencies
- **Uses**: `jax`, `jaxlib`, `equinox`, `jaxtyping`, `lineax`, `optimistix`.
- **Boundary**:
  - `src/glmax/data.py` owns the `GLMData` noun contract and input canonicalization for `offset` and `weights`.
  - `src/glmax/glm.py` owns `GLM` (pure spec noun, `family` only) and `specify` — no fit logic, no solver.
  - `src/glmax/_fit/types.py` owns `Params`, `FitResult`, `FittedGLM`, and `AbstractFitter` (abstract base with `solver: AbstractVar`).
  - `src/glmax/_fit/fit.py` owns the public `fit` and `predict` verbs (`@eqx.filter_jit`-wrapped).
  - `src/glmax/_fit/irls.py` owns `IRLSFitter` (the default fitter) and the `irls` kernel. `IRLSFitter.__call__` is NOT JIT-safe due to Python branching on family type; `fit` is safe because `IRLSFitter` is static under JIT.
  - `src/glmax/_fit/solve.py` is the canonical home for linear solver contracts (`AbstractLinearSolver`, `CholeskySolver`, `QRSolver`, `CGSolver`).
  - `src/glmax/_fit/__init__.py` re-exports all fit internals.
  - `src/glmax/diagnostics.py` owns `Diagnostics` and `check`.
  - `src/glmax/_infer/infer.py` owns `infer`.
  - `src/glmax/_infer/types.py` owns `InferenceResult`.
  - `src/glmax/_infer/hyptest.py` owns `AbstractTest`, `WaldTest`, `ScoreTest`.
  - `src/glmax/_infer/stderr.py` owns `AbstractStdErrEstimator`, `FisherInfoError`, `HuberError`.
  - `src/glmax/_infer/__init__.py` re-exports all infer internals.
  - Other modules under `src/glmax/_infer/` are internal numerics seams, not package-root API.

## Invariants
- Contract carrier split is deliberate: `GLMData`, `FitResult`, and `FittedGLM` are `equinox.Module` types with constructor-time validation; `Params`, `Diagnostics`, and `InferenceResult` are `NamedTuple` pytrees.
- Public exports stay centralized in `src/glmax/__init__.py`; if a user-facing noun or verb changes, update docs and contract tests in the same patch.
- Keep examples and documentation on the grammar nouns and top-level verbs, not stale internal import paths.
- `site/` is generated documentation output. Edit `docs/` and `mkdocs.yml`, not `site/`.

## Verification
- Do not run bare `pytest`.
- Use `pytest -p no:capture tests` for all pytest invocations.
- Keep `README.md`, `docs/index.md`, and `docs/api/glm.md` aligned with the package-root exports.
- Contract changes should update the owning tests, especially `tests/package/test_fit_api.py`, `tests/package/test_grammar_contracts.py`, and the relevant verb-specific suites.
- Public numerics and solver docstrings use raw markdown section labels: `**Arguments:**`, `**Returns:**`, and `**Raises:**` or `**Failure Modes:**`.
