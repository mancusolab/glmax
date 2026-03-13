# glmax

## Purpose
`glmax` provides grammar-first generalized linear modeling in JAX. Keep the public surface centered on explicit nouns (`GLMData`, `Params`, `FitResult`, `InferenceResult`, `Diagnostics`) and verbs (`specify`, `fit`, `predict`, `infer`, `check`) rather than wrapper-heavy or module-internal APIs.

## Contracts
- **Exposes**:
  - Package-root API from `src/glmax/__init__.py`: `GLMData`, `Params`, `GLM`, `Fitter`, `FitResult`, `InferenceResult`, `Diagnostics`, `specify`, `predict`, `fit`, `infer`, `check`.
  - Family and link implementations from `src/glmax/family/__init__.py`.
  - User-facing grammar docs in `README.md`, `docs/index.md`, and `docs/api/glm.md`.
- **Guarantees**:
  - Canonical user workflow is `specify -> fit -> predict -> infer -> check`.
  - `glmax.fit(model, data, init=None, *, fitter=...)` is the curated public fit contract.
  - `infer(model, fit_result)` and `check(model, fit_result)` operate on fit artifacts without refitting.
  - `GLM` is a pure specification noun (`family`, `solver` fields only). It has no `.fit` method. Use `glmax.fit(model, data)`.
  - `Gamma` is a supported family (exported from `glmax.family`). Dispersion estimation for Gamma is deferred.
- **Expects**:
  - `GLMData.X` is rank-2 with shape `(n, p)` and `GLMData.y` is rank-1 with shape `(n,)`.
  - Optional `offset`, `weights`, and `mask` inputs broadcast over the sample axis when present.
  - `Params.beta` is a finite inexact rank-1 vector of length `p`; `Params.disp` is a finite inexact scalar.
  - `FitResult` artifacts stay shape-aligned with `Params.beta` and the effective sample count, and invalid artifacts fail deterministic validation.
  - `GLMData.weights` is not part of the supported public `fit` or `predict` contract unless docs and tests are updated in the same change.

## Dependencies
- **Uses**: `jax`, `jaxlib`, `equinox`, `jaxtyping`, `lineax`, `optimistix`.
- **Boundary**:
  - `src/glmax/contracts.py` owns canonical noun contracts and fit-result validation.
  - `src/glmax/fit.py` owns the public `fit` and `predict` verbs.
  - `src/glmax/glm.py` owns `GLM` (pure spec noun) and `specify` only — no fit logic.
  - `src/glmax/fit.py` owns `IRLSFitter` (the default fitter), `fit`, `predict`, and `Params`/`FitResult` contracts. `IRLSFitter` is not JIT-safe; do not wrap it in `jax.jit`.
  - `src/glmax/infer/inference.py` owns `InferenceResult`, `infer`, and the standalone `wald_test` function.
  - `src/glmax/infer/__init__.py` only re-exports `infer` and `check`.
  - `src/glmax/infer/solve.py` is the canonical home for linear solver contracts.
  - Other modules under `src/glmax/infer/` are internal numerics seams, not package-root API.

## Invariants
- Contract carrier split is deliberate: `GLMData` and `FitResult` are `equinox.Module` types with constructor-time validation; `Params`, `Diagnostics`, and `InferenceResult` are `NamedTuple` pytrees.
- Public exports stay centralized in `src/glmax/__init__.py`; if a user-facing noun or verb changes, update docs and contract tests in the same patch.
- Keep examples and documentation on the grammar nouns and top-level verbs, not stale internal import paths.
- `site/` is generated documentation output. Edit `docs/` and `mkdocs.yml`, not `site/`.

## Verification
- Do not run bare `pytest`.
- Use `pytest -p no:capture ...` for all pytest invocations.
- Keep `README.md`, `docs/index.md`, and `docs/api/glm.md` aligned with the package-root exports.
- Contract changes should update the owning tests, especially `tests/test_fit_api.py`, `tests/test_grammar_contracts.py`, and the relevant verb-specific suites.
- Public numerics and solver docstrings use raw markdown section labels: `**Arguments:**`, `**Returns:**`, and `**Raises:**` or `**Failure Modes:**`.
