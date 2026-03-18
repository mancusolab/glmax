# Params Disp Aux Test Requirements

**Design:** `/Users/nicholas/Projects/glmax/.worktrees/params-disp-aux/.plans/design-plans/2026-03-17-params-disp-aux.md`

**Implementation Plan Directory:** `/Users/nicholas/Projects/glmax/.worktrees/params-disp-aux/.plans/implementation-plans/2026-03-17-params-disp-aux`

**Generated:** 2026-03-17 12:12:51 PDT

---

## Automated Test Requirements

| Acceptance Criterion | Test Type | Expected Test File Path(s) | Requirement |
| --- | --- | --- | --- |
| `params-disp-aux.AC1.1` | contract | `tests/package/test_api.py`, `tests/data/test_glmdata.py`, `tests/fit/test_fit.py` | `Params` stores `beta`, `disp`, and `aux`, remains a `NamedTuple` pytree, and survives warm-start use through `fit(...)`. |
| `params-disp-aux.AC1.2` | contract | `tests/package/test_api.py`, `tests/package/test_grammar.py` | Contract tests pin the meanings of `beta`, `disp`, and `aux`. |
| `params-disp-aux.AC1.3` | boundary | `tests/package/test_api.py`, `tests/fit/test_predict.py`, `tests/fit/test_fit.py` | Invalid `disp`/`aux` dtypes and shapes raise deterministic public-boundary errors. |
| `params-disp-aux.AC1.4` | unit | `tests/glm/test_glm.py` | Families without auxiliary state ignore provided `aux` through the `GLM` boundary and canonicalize it to `None`. |
| `params-disp-aux.AC1.5` | integration | `tests/fit/test_fit.py`, `tests/infer/test_infer.py` | Canonical warm starts preserve `disp` and `aux` through `fit(...)` and `infer(...)`. |
| `params-disp-aux.AC2.1` | numerics | `tests/family/test_families.py` | Gaussian and Gamma use `disp` as EDM dispersion and ignore `aux`. |
| `params-disp-aux.AC2.2` | numerics | `tests/family/test_families.py` | Poisson and Binomial canonicalize `disp` to `1.0` and ignore `aux`. |
| `params-disp-aux.AC2.3` | numerics/integration | `tests/family/test_families.py`, `tests/glm/test_glm.py`, `tests/fit/test_fit.py` | Negative Binomial canonicalizes `disp` to `1.0` and uses `aux` as `alpha` in likelihood, variance, sampling, and fitting updates. |
| `params-disp-aux.AC2.4` | boundary | `tests/family/test_families.py` | Non-finite or non-positive Negative Binomial `aux` values are rejected deterministically. |
| `params-disp-aux.AC2.5` | unit | `tests/glm/test_glm.py` | GLM-level tests pin the family/GLM docstring contract around which parameter each family uses. |
| `params-disp-aux.AC3.1` | integration | `tests/fit/test_fit.py`, `tests/package/test_grammar.py` | `fit(...)` returns canonical `Params(beta, disp, aux)` for every supported family. |
| `params-disp-aux.AC3.2` | inference | `tests/infer/test_stderr.py`, `tests/infer/test_hypothesis.py` | Inference reads `fitted.params.disp` as `phi` and does not treat NB `aux` as GLM dispersion. |
| `params-disp-aux.AC3.3` | integration | `tests/fit/test_predict.py`, `tests/glm/test_glm.py` | `predict(...)` and GLM mean computations remain correct under the updated carrier. |
| `params-disp-aux.AC3.4` | inference | `tests/infer/test_infer.py`, `tests/infer/test_stderr.py`, `tests/infer/test_hypothesis.py` | Wald, Score, Fisher, and Huber outputs remain finite and shape-aligned. |
| `params-disp-aux.AC3.5` | regression | `tests/fit/test_fit.py`, `tests/glm/test_glm.py`, `tests/family/test_families.py` | Supported-family convergence and Statsmodels parity checks continue to pass after the split. |

## Human Verification Requirements

| Acceptance Criterion | Why Human Review Is Needed | Files To Review | Verification Approach |
| --- | --- | --- | --- |
| `params-disp-aux.AC4.1` | The repository does not currently have documentation text assertions; semantic wording must be reviewed directly even after the docs build passes. | `README.md`, `docs/index.md`, `docs/api/nouns.md`, `docs/api/verbs.md`, `docs/api/inference.md`, `docs/api/family.md` | Confirm every public description says `disp` is GLM dispersion and `aux` is family-specific. |
| `params-disp-aux.AC4.2` | Negative Binomial terminology is prose-heavy and must be checked for statistical meaning, not just file existence. | `docs/api/family.md`, `README.md`, `AGENTS.md` | Confirm NB `alpha` is documented as `aux`, never as `params.disp`. |
| `params-disp-aux.AC4.3` | Removing stale semantics requires a repo-wide wording review in addition to test updates. | `AGENTS.md`, `tests/glm/test_glm.py`, `tests/fit/test_fit.py`, `tests/infer/test_stderr.py` | Confirm no remaining references describe NB `params.disp` as `alpha`. |
| `params-disp-aux.AC4.4` | Workflow emphasis is a documentation judgment call that is not captured by automated tests. | `README.md`, `docs/index.md`, `docs/api/fitters.md`, `docs/api/inference.md` | Confirm advanced `_fit` details remain secondary and the top-level grammar workflow stays primary. |

## Operational Checks

- Run `pytest -p no:capture tests/package/test_api.py tests/package/test_grammar.py tests/fit tests/infer tests/glm tests/family tests/data` after Phase 4 to ensure the full regression suite matches the new carrier semantics.
- Run `mkdocs build --strict` after Phase 4 to ensure the newly created `docs/api/*.md` files satisfy the navigation already declared in `mkdocs.yml`.
