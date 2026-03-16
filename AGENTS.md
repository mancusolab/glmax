# glmax

## Purpose
`glmax` provides grammar-first generalized linear modeling in JAX. Keep the public surface centered on explicit nouns (`GLMData`, `Params`, `FitResult`, `FittedGLM`, `InferenceResult`, `Diagnostics`) and verbs (`specify`, `fit`, `predict`, `infer`, `check`) rather than wrapper-heavy or module-internal APIs.

## Contracts
- **Exposes**:
  - Package-root API from `src/glmax/__init__.py`: `GLMData`, `Params`, `GLM`, `Fitter`, `FitResult`, `FittedGLM`, `InferenceResult`, `Diagnostics`, `specify`, `predict`, `fit`, `infer`, `check`.
  - Family and link implementations from `src/glmax/family/__init__.py`.
  - User-facing grammar docs in `README.md`, `docs/index.md`, and `docs/api/glm.md`.
- **Guarantees**:
  - Canonical user workflow is `specify -> fit -> predict -> infer -> check`.
  - `glmax.fit(model, data, init=None, *, fitter=...)` is the curated public fit contract and returns `FittedGLM`.
  - `infer(fitted, stderr=...)` and `check(fitted)` operate on the fitted noun without refitting.
  - `GLM` is a pure specification noun (`family`, `solver` fields only). It has no `.fit` method. Use `glmax.fit(model, data)`.
  - `Gamma` is a supported family (exported from `glmax.family`). Dispersion estimation for Gamma is deferred.
- **Expects**:
  - `GLMData.X` is rank-2 with shape `(n, p)` and `GLMData.y` is rank-1 with shape `(n,)`.
  - Optional `offset`, `weights`, and `mask` inputs broadcast over the sample axis when present.
  - `Params.beta` is a finite inexact rank-1 vector of length `p`; `Params.disp` is a finite inexact scalar.
  - `FitResult` is the fitter contract and carries `params`, `X`, `y`, `eta`, `mu`, `glm_wt`, `converged`, `num_iters`, `objective`, `objective_delta`, and `score_residual`.
  - `FittedGLM` is the public fitted noun and binds `model` plus `result`, forwarding common fit artifacts for ergonomics.
  - `InferenceResult` carries `params`, `se`, `z`, and `p`; those summaries are produced by `infer(fitted, stderr=...)`, not `fit(...)`.
  - `Diagnostics` is reserved for `check(fitted)` model-fit assessment. The current `check()` seam is a placeholder contract and does not yet expose finalized model-diagnostic fields.
  - `FitResult` artifacts stay shape-aligned with `Params.beta` and the effective sample count, and invalid artifacts fail deterministic validation.
  - `GLMData.weights` is not part of the supported public `fit` or `predict` contract unless docs and tests are updated in the same change.

## Dependencies
- **Uses**: `jax`, `jaxlib`, `equinox`, `jaxtyping`, `lineax`, `optimistix`.
- **Boundary**:
  - `src/glmax/data.py` owns the `GLMData` noun contract and input canonicalization for `offset`, `weights`, and `mask`.
  - `src/glmax/glm.py` owns `GLM` (pure spec noun) and `specify` only — no fit logic.
  - `src/glmax/fit.py` owns `Params`, `FitResult`, `FittedGLM`, fit-result validation, `IRLSFitter` (the default fitter), and the public `fit` / `predict` verbs. `IRLSFitter` is not JIT-safe; do not wrap it in `jax.jit`.
  - `src/glmax/infer/diagnostics.py` owns `Diagnostics` and `check`.
  - `src/glmax/infer/inference.py` owns `InferenceResult`, `infer`, and the standalone `wald_test` function.
  - `src/glmax/infer/__init__.py` only re-exports `infer` and `check`.
  - `src/glmax/infer/solve.py` is the canonical home for linear solver contracts.
  - Other modules under `src/glmax/infer/` are internal numerics seams, not package-root API.

## Invariants
- Contract carrier split is deliberate: `GLMData`, `FitResult`, and `FittedGLM` are `equinox.Module` types with constructor-time validation; `Params`, `Diagnostics`, and `InferenceResult` are `NamedTuple` pytrees.
- Public exports stay centralized in `src/glmax/__init__.py`; if a user-facing noun or verb changes, update docs and contract tests in the same patch.
- Keep examples and documentation on the grammar nouns and top-level verbs, not stale internal import paths.
- `site/` is generated documentation output. Edit `docs/` and `mkdocs.yml`, not `site/`.

## Verification
- Do not run bare `pytest`.
- Use `pytest -p no:capture ...` for all pytest invocations.
- Keep `README.md`, `docs/index.md`, and `docs/api/glm.md` aligned with the package-root exports.
- Contract changes should update the owning tests, especially `tests/test_fit_api.py`, `tests/test_grammar_contracts.py`, and the relevant verb-specific suites.
- Public numerics and solver docstrings use raw markdown section labels: `**Arguments:**`, `**Returns:**`, and `**Raises:**` or `**Failure Modes:**`.
