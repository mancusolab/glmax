# expfam-port Implementation Plan — Phase 6: Test Updates and Verification

**Goal:** Update test suite to reflect the changes in Phases 1–5, fix the normalization bug in `test_glm.py`, and run the full suite to verify green.

**Architecture:** Four sets of changes: (1) one-line normalization fix in `test_glm.py`, (2) replace all `model.fit(data)` and `GLM(…).fit(data)` calls in `test_boundaries.py` with `glmax.fit(…)` and update Gaussian dispersion assertions, (3) update `test_fit_api.py` to expect `AttributeError` from the now-removed `GLM.fit` attribute and fix the Gaussian dispersion assertion, (4) run and commit. No new files.

**Tech Stack:** Python, pytest, JAX

**Scope:** Phase 6 of 6

**Codebase verified:** 2026-03-09

---

## Acceptance Criteria Coverage

### expfam-port.AC2: Family module improved

- **expfam-port.AC2.5 (success):** After fitting a Gaussian GLM, `result.params.disp > 0`.

### expfam-port.AC3: GLM internals cleaned

- **expfam-port.AC3.1 (success):** `hasattr(GLM(), 'fit')` returns `False`.
- **expfam-port.AC3.7 (failure):** `model.fit(data)` raises `AttributeError`.

### expfam-port.AC4: Tests pass

- **expfam-port.AC4.1 (success):** `pytest -p no:capture tests` exits with code 0, zero failures, zero errors.
- **expfam-port.AC4.2 (success):** Gaussian standardization in `test_glm.py` uses `(X - mean) / std` parenthesization.

---

<!-- START_SUBCOMPONENT_A (tasks 1-4) -->

<!-- START_TASK_1 -->
### Task 1: Fix normalization bug in `test_glm.py`

**Verifies:** expfam-port.AC4.2

**Files:**
- Modify: `tests/test_glm.py:54`

**Step 1: Fix the one-line bug**

At line 54, the current code is:

```python
    X = X - X.mean(axis=0) / (X.std(axis=0))
```

Due to operator precedence this subtracts `mean/std` from `X` rather than standardizing. Replace with:

```python
    X = (X - X.mean(axis=0)) / X.std(axis=0)
```

No other changes to `test_glm.py`.
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Update `test_boundaries.py`

**Verifies:** expfam-port.AC2.5, expfam-port.AC3.7

**Files:**
- Modify: `tests/test_boundaries.py`

After Phase 5, `GLM.fit` no longer exists. All `GLM(…).fit(data)` and `model.fit(data)` calls must be replaced with `glmax.fit(model, data)`. Gaussian `Params.disp` is now `sigma^2 > 0` (Phase 3), so the fixed-dispersion parametrized test needs a Gaussian-specific branch.

**Step 1: Add `Params` to imports**

Current import line:

```python
from glmax import GLM, GLMData
```

Replace with:

```python
from glmax import GLM, GLMData, Params
```

**Step 2: Fix `test_glm_fit_accepts_glmdata_noun` (line 69–72)**

```python
def test_glm_fit_accepts_glmdata_noun() -> None:
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=jnp.array([0.1, 1.0, 2.0, 2.9]))
    fit_result = glmax.fit(GLM(family=Gaussian()), data)
    assert fit_result.params.beta.shape == (1,)
```

**Step 3: Fix `test_glm_fit_accepts_canonical_disp_init_keyword` (lines 80–86)**

The `disp_init=` keyword is gone. The new API uses `init=Params(beta, disp)`. Rename the function to reflect the new pattern and replace the call:

```python
def test_glmax_fit_accepts_params_init_for_nb() -> None:
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=jnp.array([0.0, 1.0, 1.0, 2.0]))
    model = GLM(family=NegativeBinomial())

    fit_result = glmax.fit(model, data, init=Params(beta=jnp.zeros(1), disp=jnp.array(0.4)))

    assert fit_result.params.beta.shape == (1,)
```

Note: this test is intentionally NB-only. The original test checked NB because `disp_init` was meaningful for NB overdispersion. Gaussian `init` passing is covered by `test_fit_boundary_rejects_non_finite_params_init` which already uses `init=Params(...)` directly.

**Step 4: Fix `test_glm_fit_rejects_all_false_mask_with_deterministic_error` (lines 89–93)**

```python
def test_glm_fit_rejects_all_false_mask_with_deterministic_error() -> None:
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0]]), y=jnp.array([0.0, 1.0, 2.0]), mask=False)

    with pytest.raises(ValueError, match="mask removes all samples"):
        glmax.fit(GLM(family=Gaussian()), data)
```

**Step 5: Fix `test_params_schema_is_beta_and_disp_only` (lines 96–101)**

```python
def test_params_schema_is_beta_and_disp_only() -> None:
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=jnp.array([0.1, 1.0, 2.0, 2.9]))
    fit_result = glmax.fit(GLM(family=Gaussian()), data)

    assert list(fit_result.params._fields) == ["beta", "disp"]
    assert not hasattr(fit_result, "alpha")
```

**Step 6: Fix `test_fixed_dispersion_families_emit_deterministic_disp` (lines 104–113)**

After Phase 3, Gaussian `estimate_dispersion` returns `RSS/(n-p) > 0`, so `Params.disp > 0` for Gaussian. Poisson and Binomial still have `Params.disp == 0.0`. Split the assertion:

```python
@pytest.mark.parametrize("family", [Gaussian(), Poisson(), Binomial()])
def test_fixed_dispersion_families_emit_deterministic_disp(family) -> None:
    X = jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    if isinstance(family, Binomial):
        y = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])
    else:
        y = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    fit_result = glmax.fit(GLM(family=family), GLMData(X=X, y=y))
    if isinstance(family, Gaussian):
        assert fit_result.params.disp > 0
    else:
        assert jnp.allclose(fit_result.params.disp, jnp.array(1.0))  # 1.0 = canonical phi for Poisson/Binomial after Phase 3
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Update `test_fit_api.py`

**Verifies:** expfam-port.AC2.5, expfam-port.AC3.1, expfam-port.AC3.7

**Files:**
- Modify: `tests/test_fit_api.py`

After Phase 5, `GLM.fit` is a removed attribute. Calling `model.fit(...)` raises `AttributeError`, not `TypeError`. The migration-guidance tests that expected specific `TypeError` messages no longer make sense. Simplify them to verify `AttributeError`.

**Step 0: Remove three tests that access `GLM.fit` as a descriptor (lines 241–271)**

After Phase 5, `glmax.GLM.fit` no longer exists as an attribute at all. The following three tests access it directly and will raise `AttributeError` before reaching their assertions:

- `test_glm_fit_is_not_a_curated_public_contract` (line 241): accesses `glmax.GLM.fit.__doc__`
- `test_glm_fit_signature_does_not_expose_legacy_wrapper_parameters` (line 248): calls `inspect.signature(glmax.GLM.fit)`
- `test_bound_glm_fit_signature_does_not_expose_model_parameter` (line 259): calls `inspect.signature(glmax.GLM(family=Gaussian()).fit)`

These three tests verify behaviour of the migration shim that no longer exists. Replace all three with a single test verifying AC3.1:

```python
def test_glm_fit_attribute_is_removed() -> None:
    assert not hasattr(glmax.GLM(), "fit"), "GLM.fit must not exist after Phase 5"
```

Place this replacement immediately before the `@pytest.mark.parametrize` block at line 274.

**Step 1: Fix `test_glm_fit_removed_legacy_keywords_raise_migration_typeerrors` (lines 274–286)**

The `@pytest.mark.parametrize` specifies legacy keyword names and error-message patterns that were part of the old graceful-deprecation shim. After Phase 5, `model.fit` raises `AttributeError` unconditionally. Replace the whole function:

```python
@pytest.mark.parametrize(
    "legacy_keyword",
    ["init", "alpha_init"],
)
def test_glm_fit_removed_raises_attributeerror(legacy_keyword: str) -> None:
    model = glmax.GLM(family=Gaussian())
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=jnp.array([0.0, 1.0, 2.0, 3.0]))

    with pytest.raises(AttributeError):
        model.fit(data, **{legacy_keyword: jnp.zeros(1)})
```

**Step 2: Fix `test_glm_fit_rejects_legacy_extra_positional_arguments_with_migration_guidance` (lines 289–309)**

Same reasoning — replace with an `AttributeError` check:

```python
def test_glm_fit_removed_raises_attributeerror_on_positional_arg() -> None:
    model = glmax.GLM(family=Gaussian())
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=jnp.array([0.0, 1.0, 2.0, 3.0]))

    with pytest.raises(AttributeError):
        model.fit(data, jnp.zeros(4))
```

**Step 3: Fix `test_canonical_fit_succeeds_for_supported_families` dispersion assertion (lines 351–354)**

After Phase 3, Gaussian `Params.disp = sigma^2 > 0`. Add Gaussian to the "dispersion is positive" branch:

```python
    if isinstance(family, (NegativeBinomial, Gaussian)):
        assert result.params.disp > 0
    else:
        assert jnp.allclose(result.params.disp, jnp.array(0.0))
```

**Step 4: Verify no remaining `model.fit(` calls in tests**

Run this grep across ALL test files to confirm no old `.fit(` calls on model objects remain (statsmodels `.fit()` calls inside `test_glm.py` are OK — they have no model variable prefix):

```bash
grep -rn "model\.fit\b\|GLM(.*\.fit\b" tests/
```

Expected: zero matches (statsmodels calls like `sm_poi.fit()` use different variable names and will not match). If any non-statsmodels `.fit()` calls appear, fix them before proceeding to Task 4.
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Run full test suite and commit

**Verifies:** expfam-port.AC4.1

**Files:**
- Test: `tests/`

**Step 1: Run the full test suite**

```bash
cd /Users/nicholas/Projects/glmax/.worktrees/expfam-port
pytest -p no:capture tests
```

Expected: all tests pass (87 existing + any new tests from Phases 3–4), 0 failures, 0 errors. If any test fails, diagnose and fix before proceeding.

**Step 2: Spot-check the ACs**

Run these one-liners to confirm key acceptance criteria:

```bash
# AC3.1: GLM has no .fit attribute
python -c "import glmax; g = glmax.GLM(); print('fit absent:', not hasattr(g, 'fit'))"

# AC2.5: Gaussian disp > 0 after fitting
python -c "
import jax.numpy as jnp
import glmax
from glmax import GLMData
data = GLMData(X=jnp.array([[0.0],[1.0],[2.0],[3.0]]), y=jnp.array([0.1,1.0,2.0,2.9]))
r = glmax.fit(glmax.GLM(), data)
print('Gaussian disp > 0:', float(r.params.disp) > 0)
"

# AC3.5: wald_test importable from _infer.inference
python -c "from glmax.infer.inference import wald_test; print('wald_test:', wald_test)"
```

Expected: all three print True/a function object without errors.

**Step 3: Commit**

```bash
git add tests/test_glm.py tests/test_boundaries.py tests/test_fit_api.py
git commit -m "test: update test suite for Phase 1-5 changes"
```
<!-- END_TASK_4 -->

<!-- END_SUBCOMPONENT_A -->
