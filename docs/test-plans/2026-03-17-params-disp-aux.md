# Params Disp Aux Human Test Plan

## Preconditions

- Use worktree `/Users/nicholas/Projects/glmax/.worktrees/params-disp-aux` at
  `HEAD=0511193`.
- Ensure the `jax` environment used for verification is available.
- Have the generated docs available from `mkdocs build --strict` and the source
  files open for side-by-side review.

## Phase Checks

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Review `tests/package/test_api.py`, `tests/package/test_grammar.py`, `tests/data/test_glmdata.py`, and `src/glmax/_fit/types.py`. | The carrier is consistently `Params(beta, disp, aux)`; no remaining two-field assumptions; public-boundary validation still targets `disp` and `aux` separately. |
| 2 | Review `src/glmax/family/dist.py`, `src/glmax/glm.py`, `tests/family/test_families.py`, and `tests/glm/test_glm.py`. | Gaussian/Gamma use `disp`; Poisson/Binomial fix `disp=1.0`; Negative Binomial uses `aux` as `alpha`; GLM delegates the split contract consistently. |
| 3 | Review `src/glmax/_fit/irls.py`, `src/glmax/_infer/stderr.py`, `src/glmax/_infer/hyptest.py`, `tests/fit/test_fit.py`, and `tests/infer/*.py`. | Fit artifacts, inference covariance scaling, and final NB `glm_wt` all align with canonical `Params(beta, disp, aux)` semantics. |
| 4 | Review `README.md`, `docs/index.md`, `docs/api/nouns.md`, `docs/api/verbs.md`, `docs/api/fitters.md`, `docs/api/inference.md`, `docs/api/family.md`, and `AGENTS.md`. | Public terminology says `disp` is GLM dispersion and `aux` is family-specific state; Negative Binomial `alpha` is documented only as `aux`; the top-level grammar workflow remains primary. |

## End-to-End Scenarios

| Scenario | Steps | Expected Result |
|----------|-------|-----------------|
| Gaussian workflow sanity | In a REPL, run `specify -> fit -> predict -> infer -> check` on a small Gaussian dataset. | `fitted.params.disp` is positive, `fitted.params.aux is None`, predictions are finite, inference outputs are finite and shape-aligned, diagnostics call returns without refit. |
| Negative Binomial workflow sanity | Fit a small Negative Binomial model, inspect `fitted.params`, `fitted.glm_wt`, and run `infer(fitted)`. | `fitted.params.disp == 1.0`, `fitted.params.aux > 0`, `fitted.glm_wt` matches `model.working_weights(fitted.eta, fitted.params.disp, fitted.params.aux)[2]`, and inference remains finite without treating `aux` as `phi`. |
| Documentation/navigation sanity | Open the built docs landing page and each `docs/api/*.md` target, then compare against source markdown. | All API pages exist, nav targets resolve, terminology matches the implemented contract, and advanced `_fit` details remain secondary to the package-root workflow. |

## Traceability

| AC ID | Automated Evidence | Human Step |
|------|---------------------|------------|
| `params-disp-aux.AC1.*` | Full regression command; package/data/fit contract suites pass. | Phase Check 1 |
| `params-disp-aux.AC2.*` | Full regression command; family/GLM suites pass. | Phase Check 2 |
| `params-disp-aux.AC3.*` | Full regression command; fit/infer/GLM suites pass. | Phase Check 3 |
| `params-disp-aux.AC4.1` | `mkdocs build --strict` plus source docs present. | Phase Check 4 |
| `params-disp-aux.AC4.2` | `mkdocs build --strict`; docs and AGENTS sources present. | Phase Check 4 |
| `params-disp-aux.AC4.3` | Full regression command plus source docs/tests present. | Phase Check 4 |
| `params-disp-aux.AC4.4` | `mkdocs build --strict` plus source docs present. | Phase Check 4 |
