# GLM Diagnostics Implementation Plan — Phase 1

**Goal:** Add `cdf` and `deviance_contribs` methods to `ExponentialDispersionFamily` and all five concrete families.

**Architecture:** Two new abstract methods on the base class; each family implements them using the same parameterisation already used in `negloglikelihood` and `sample`. No new files needed.

**Tech Stack:** JAX, Equinox, `jax.scipy.stats` (already imported in dist.py)

**Scope:** Phase 1 of 4

**Codebase verified:** 2026-03-19

---

## Acceptance Criteria Coverage

### glm-diagnostics.AC3: DevianceResidual
- **glm-diagnostics.AC3.2 Edge:** Poisson with `y=0` produces a finite (non-NaN) value using `0 * log(0/mu) = 0` convention

*(AC3.2 requires `deviance_contribs` on Poisson to handle y=0 correctly — this is the Phase 1 prerequisite.)*

---

## Key Codebase Facts

- Abstract base: `ExponentialDispersionFamily` in `src/glmax/family/dist.py:48`
- No `strict=True` on family classes
- Existing abstract methods: `negloglikelihood(y, eta, disp, aux)`, `variance(mu, disp, aux)`, `sample(key, eta, disp, aux)`
- Parameterisations verified:
  - **Gaussian**: `disp` = σ², CDF via `jaxstats.norm.cdf(y, mu, sqrt(disp))`; safe_disp sentinel present in existing methods
  - **Poisson**: `disp` ignored; CDF via `jaxstats.poisson.cdf(y, mu)`
  - **Binomial**: `disp` ignored (Bernoulli); CDF via `jaxstats.bernoulli.cdf(y, mu)`
  - **Gamma**: `disp` = φ, shape k=1/φ, scale=μφ; CDF via `jaxstats.gamma.cdf(y, a=k, scale=theta)`
  - **NegativeBinomial**: alpha from `_nb_alpha_from_split(disp, aux)`; no jaxstats.nbinom — must implement manually via `jax.scipy.special`
- `jax.scipy.stats as jaxstats` already imported; `gammaln` already imported from `jax.scipy.special`
- Add `xlogy` to the existing `jax.scipy.special` import: `from jax.scipy.special import gammaln, xlogy`
- `xlogy(x, y)` computes `x * log(y)` with `xlogy(0, 0) == 0`; use for all `y * log(y/mu)` terms
- `jnp.log1p` is used only where the argument is computed directly without cancellation: NegBin r-term uses `log1p((mu-y)/(r+y))`; Gamma uses `log1p(r_)` where `r_ = (y-mu)/mu` computed directly (not as `y/mu - 1`)
- Test file: `tests/family/test_families.py` — uses `pytest.mark.parametrize`, no statsmodels
- Run tests with: `pytest -p no:capture tests`

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->

<!-- START_TASK_1 -->
### Task 1: Add `cdf` and `deviance_contribs` abstract methods to `ExponentialDispersionFamily`

**Verifies:** Prerequisite for all Phase 2–3 ACs

**Files:**
- Modify: `src/glmax/family/dist.py:88` (after the `variance` abstract method, before `sample`)

**Implementation:**

Insert two new abstract methods after `variance` (line ~101) and before `sample` (line ~103):

```python
@abstractmethod
def cdf(
    self,
    y: ArrayLike,
    mu: ArrayLike,
    disp: ScalarLike = 0.0,
    aux: ScalarLike | None = None,
) -> Array:
    r"""Cumulative distribution function $F(y \mid \mu)$.

    **Arguments:**

    - `y`: observed responses, shape `(n,)`.
    - `mu`: fitted means, shape `(n,)`.
    - `disp`: dispersion parameter, scalar.
    - `aux`: optional family-specific auxiliary scalar.

    **Returns:**

    CDF values $F(y_i \mid \mu_i)$, shape `(n,)`, values in `[0, 1]`.
    """

@abstractmethod
def deviance_contribs(
    self,
    y: ArrayLike,
    mu: ArrayLike,
    disp: ScalarLike = 0.0,
    aux: ScalarLike | None = None,
) -> Array:
    r"""Per-observation deviance contributions $d_i = 2(\ell_i^\text{sat} - \ell_i^\text{fit})$.

    **Arguments:**

    - `y`: observed responses, shape `(n,)`.
    - `mu`: fitted means, shape `(n,)`.
    - `disp`: dispersion parameter, scalar.
    - `aux`: optional family-specific auxiliary scalar.

    **Returns:**

    Non-negative deviance contributions, shape `(n,)`.
    """
```

**Verification:**

Run: `pytest -p no:capture tests`
Expected: 321 existing tests still pass (no new tests yet; implementations in Tasks 2 and 3)

**Commit:** `feat(family): add cdf and deviance_contribs abstract methods to ExponentialDispersionFamily`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Implement `cdf` and `deviance_contribs` for Gaussian and Poisson

**Verifies:** Prerequisite for glm-diagnostics.AC2, AC3, AC3.2, AC4, AC5

**Files:**
- Modify: `src/glmax/family/dist.py` — Gaussian class (~line 200), Poisson class (~line 406)
- Test: `tests/family/test_families.py`

**Implementation:**

**Gaussian** (insert after the `sample` method, ~line 324):

```python
def cdf(
    self,
    y: ArrayLike,
    mu: ArrayLike,
    disp: ScalarLike = 1.0,
    aux: ScalarLike | None = None,
) -> Array:
    r"""Gaussian CDF $\Phi\!\left(\frac{y - \mu}{\sigma}\right)$.

    **Arguments:**

    - `y`: observed responses, shape `(n,)`.
    - `mu`: fitted means, shape `(n,)`.
    - `disp`: variance $\sigma^2$, scalar.
    - `aux`: ignored.

    **Returns:**

    CDF values, shape `(n,)`.
    """
    del aux
    safe_disp = jnp.where(jnp.asarray(disp) > 0, disp, 1.0)
    return jaxstats.norm.cdf(y, loc=mu, scale=jnp.sqrt(safe_disp))

def deviance_contribs(
    self,
    y: ArrayLike,
    mu: ArrayLike,
    disp: ScalarLike = 1.0,
    aux: ScalarLike | None = None,
) -> Array:
    r"""Gaussian deviance contributions $(y_i - \mu_i)^2$.

    **Arguments:**

    - `y`: observed responses, shape `(n,)`.
    - `mu`: fitted means, shape `(n,)`.
    - `disp`: ignored for unscaled deviance.
    - `aux`: ignored.

    **Returns:**

    Non-negative deviance contributions, shape `(n,)`.
    """
    del disp, aux
    return (jnp.asarray(y) - jnp.asarray(mu)) ** 2
```

**Import update** — add `xlogy` to the existing `jax.scipy.special` import at the top of `dist.py`:
```python
from jax.scipy.special import gammaln, xlogy
```

**Poisson** (insert after the `sample` method, ~line 475):

```python
def cdf(
    self,
    y: ArrayLike,
    mu: ArrayLike,
    disp: ScalarLike = 0.0,
    aux: ScalarLike | None = None,
) -> Array:
    r"""Poisson CDF $\sum_{k=0}^{\lfloor y \rfloor} e^{-\mu} \mu^k / k!$.

    **Arguments:**

    - `y`: count responses, shape `(n,)`.
    - `mu`: fitted means (rates), shape `(n,)`.
    - `disp`: ignored.
    - `aux`: ignored.

    **Returns:**

    CDF values, shape `(n,)`.
    """
    del disp, aux
    return jaxstats.poisson.cdf(y, mu=jnp.asarray(mu))

def deviance_contribs(
    self,
    y: ArrayLike,
    mu: ArrayLike,
    disp: ScalarLike = 0.0,
    aux: ScalarLike | None = None,
) -> Array:
    r"""Poisson deviance contributions $2(y_i \log(y_i/\mu_i) - (y_i - \mu_i))$.

    Uses `xlogy` for correct handling of $y_i = 0$ via `xlogy(0, 0) = 0`.

    **Arguments:**

    - `y`: count responses, shape `(n,)`.
    - `mu`: fitted means, shape `(n,)`.
    - `disp`: ignored.
    - `aux`: ignored.

    **Returns:**

    Non-negative deviance contributions, shape `(n,)`.
    """
    del disp, aux
    y_ = jnp.asarray(y)
    mu_ = jnp.asarray(mu)
    return 2.0 * (xlogy(y_, y_ / mu_) - (y_ - mu_))
```

**Testing:**

Add to `tests/family/test_families.py`:

```python
import scipy.stats

class TestCdf:
    def test_gaussian_cdf_matches_scipy(self):
        f = Gaussian()
        y = jnp.array([0.0, 1.0, 2.0, -1.0])
        mu = jnp.array([0.5, 1.5, 1.0, 0.0])
        disp = 2.0
        result = f.cdf(y, mu, disp)
        expected = scipy.stats.norm.cdf(y, loc=mu, scale=jnp.sqrt(disp))
        assert jnp.allclose(result, expected, atol=1e-10)

    def test_poisson_cdf_matches_scipy(self):
        f = Poisson()
        y = jnp.array([0.0, 1.0, 3.0, 5.0])
        mu = jnp.array([1.0, 2.0, 2.5, 4.0])
        result = f.cdf(y, mu)
        expected = scipy.stats.poisson.cdf(y, mu=mu)
        assert jnp.allclose(result, expected, atol=1e-10)


class TestDevianceContribs:
    def test_gaussian_deviance_contribs_matches_formula(self):
        f = Gaussian()
        y = jnp.array([1.0, 2.0, 3.0])
        mu = jnp.array([1.5, 1.8, 2.5])
        result = f.deviance_contribs(y, mu)
        expected = (y - mu) ** 2
        assert jnp.allclose(result, expected, atol=1e-12)

    def test_gaussian_deviance_sum_equals_rss(self):
        # Gaussian total deviance = RSS = sum((y-mu)^2)
        y = jnp.array([1.2, 0.8, 2.1, 1.5, 0.9])
        mu = jnp.array([1.0, 1.0, 2.0, 1.5, 1.0])
        f = Gaussian()
        rss = jnp.sum((y - mu) ** 2)
        assert jnp.allclose(jnp.sum(f.deviance_contribs(y, mu)), rss, atol=1e-10)

    def test_poisson_deviance_zero_y_is_finite(self):
        f = Poisson()
        y = jnp.array([0.0, 1.0, 2.0])
        mu = jnp.array([0.5, 1.0, 1.5])
        result = f.deviance_contribs(y, mu)
        assert jnp.all(jnp.isfinite(result))
        # y=0 term: 2*(0 - (0 - 0.5)) = 1.0
        assert jnp.allclose(result[0], 1.0, atol=1e-10)

    def test_poisson_deviance_contribs_formula(self):
        # Verify: d_i = 2*(y*log(y/mu) - (y-mu)), with 0*log(0/mu) = 0
        y = jnp.array([0.0, 1.0, 3.0, 5.0, 2.0])
        mu = jnp.array([0.5, 1.5, 2.0, 4.5, 2.5])
        f = Poisson()
        result = f.deviance_contribs(y, mu)
        log_term = jnp.where(y > 0, y * jnp.log(y / mu), 0.0)
        expected = 2.0 * (log_term - (y - mu))
        assert jnp.allclose(result, expected, atol=1e-10)

    @pytest.mark.parametrize("FamilyCls", [Gaussian, Poisson])
    def test_deviance_contribs_nonnegative(self, FamilyCls):
        f = FamilyCls()
        y = jnp.array([1.0, 2.0, 0.5])
        mu = jnp.array([1.0, 2.0, 0.5])
        # When y == mu, deviance_contribs should be 0
        result = f.deviance_contribs(y, mu)
        assert jnp.all(result >= 0)
        assert jnp.allclose(result, 0.0, atol=1e-10)
```

**Verification:**

Run: `pytest -p no:capture tests/family/`
Expected: All tests pass

**Commit:** `feat(family): implement cdf and deviance_contribs for Gaussian and Poisson`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Implement `cdf` and `deviance_contribs` for Binomial, Gamma, and NegativeBinomial

**Verifies:** Prerequisite for glm-diagnostics.AC4 (QuantileResidual for all families)

**Files:**
- Modify: `src/glmax/family/dist.py` — Binomial (~line 327), NegativeBinomial (~line 478), Gamma (~line 679)
- Test: `tests/family/test_families.py`

**Implementation:**

**Binomial** (insert after the `sample` method, ~line 403):

```python
def cdf(
    self,
    y: ArrayLike,
    mu: ArrayLike,
    disp: ScalarLike = 0.0,
    aux: ScalarLike | None = None,
) -> Array:
    r"""Bernoulli CDF: $F(y \mid \mu) = 1 - \mu$ for $y < 1$, else $1$.

    **Arguments:**

    - `y`: binary responses in `{0, 1}`, shape `(n,)`.
    - `mu`: success probabilities, shape `(n,)`.
    - `disp`: ignored.
    - `aux`: ignored.

    **Returns:**

    CDF values, shape `(n,)`.
    """
    del disp, aux
    return jaxstats.bernoulli.cdf(y, p=jnp.asarray(mu))

def deviance_contribs(
    self,
    y: ArrayLike,
    mu: ArrayLike,
    disp: ScalarLike = 0.0,
    aux: ScalarLike | None = None,
) -> Array:
    r"""Binomial deviance contributions $-2(y_i \log \mu_i + (1-y_i)\log(1-\mu_i))$.

    **Arguments:**

    - `y`: binary responses in `{0, 1}`, shape `(n,)`.
    - `mu`: success probabilities, shape `(n,)`.
    - `disp`: ignored.
    - `aux`: ignored.

    **Returns:**

    Non-negative deviance contributions, shape `(n,)`.
    """
    del disp, aux
    y_ = jnp.asarray(y)
    mu_ = jnp.asarray(mu)
    term1 = xlogy(y_, y_ / mu_)
    term2 = xlogy(1.0 - y_, (1.0 - y_) / (1.0 - mu_))
    return 2.0 * (term1 + term2)
```

**NegativeBinomial** (insert after the `init_nuisance` method, ~line 676):

NB-2 parameterisation: r = 1/α, p = μ/(r+μ). The CDF is `nbinom(n=r, p=r/(r+μ)).cdf(y)`. JAX does not have `jaxstats.nbinom`; use the regularised incomplete beta function via `jax.scipy.special.betainc`.

Note: `scipy.special.betainc(a, b, x) = I_x(a, b)` is the regularised incomplete beta.
`CDF_nbinom(y; r, p_fail) = I_{p_fail}(r, y+1)` where `p_fail = r/(r+μ)`.

```python
def cdf(
    self,
    y: ArrayLike,
    mu: ArrayLike,
    disp: ScalarLike = 1.0,
    aux: ScalarLike | None = None,
) -> Array:
    r"""Negative-binomial CDF via the regularised incomplete beta function.

    Uses $F(y; r, p_\text{fail}) = I_{p_\text{fail}}(r, \lfloor y \rfloor + 1)$
    where $r = 1/\alpha$ and $p_\text{fail} = r / (r + \mu)$.

    **Arguments:**

    - `y`: count responses, shape `(n,)`.
    - `mu`: fitted means, shape `(n,)`.
    - `disp`: legacy alpha carrier (used if `aux` is None).
    - `aux`: overdispersion $\alpha > 0$, scalar.

    **Returns:**

    CDF values, shape `(n,)`.
    """
    alpha = _nb_alpha_from_split(disp, aux)
    r = 1.0 / alpha
    mu_ = jnp.asarray(mu)
    p_fail = r / (r + mu_)
    return jax.scipy.special.betainc(r, jnp.floor(y) + 1.0, p_fail)

def deviance_contribs(
    self,
    y: ArrayLike,
    mu: ArrayLike,
    disp: ScalarLike = 1.0,
    aux: ScalarLike | None = None,
) -> Array:
    r"""Negative-binomial deviance contributions.

    $d_i = 2\bigl(r \log\tfrac{r+\mu_i}{r+y_i} + y_i \log\tfrac{y_i}{\mu_i}\bigr)$
    where $r = 1/\alpha$.  Uses `log1p` for the $r$-term (argument computed
    directly as $(μ-y)/(r+y)$, stable near $y \approx \mu$) and `xlogy` for
    the $y$-term.

    **Arguments:**

    - `y`: count responses, shape `(n,)`.
    - `mu`: fitted means, shape `(n,)`.
    - `disp`: legacy alpha carrier.
    - `aux`: overdispersion $\alpha > 0$, scalar.

    **Returns:**

    Non-negative deviance contributions, shape `(n,)`.
    """
    alpha = _nb_alpha_from_split(disp, aux)
    r = 1.0 / alpha
    y_ = jnp.asarray(y)
    mu_ = jnp.asarray(mu)
    return 2.0 * (r * jnp.log1p((mu_ - y_) / (r + y_)) + xlogy(y_, y_ / mu_))
```

**Gamma** (insert after the `sample` method, ~line 790):

```python
def cdf(
    self,
    y: ArrayLike,
    mu: ArrayLike,
    disp: ScalarLike = 1.0,
    aux: ScalarLike | None = None,
) -> Array:
    r"""Gamma CDF with shape $k = 1/\phi$, scale $\theta = \mu\phi$.

    **Arguments:**

    - `y`: positive responses, shape `(n,)`.
    - `mu`: fitted means, shape `(n,)`.
    - `disp`: dispersion $\phi > 0$, scalar.
    - `aux`: ignored.

    **Returns:**

    CDF values, shape `(n,)`.
    """
    del aux
    safe_disp = jnp.where(jnp.asarray(disp) > 0, disp, 1.0)
    mu_ = jnp.clip(jnp.asarray(mu), *self._bounds)
    k = 1.0 / safe_disp
    theta = mu_ * safe_disp
    return jaxstats.gamma.cdf(y, a=k, scale=theta)

def deviance_contribs(
    self,
    y: ArrayLike,
    mu: ArrayLike,
    disp: ScalarLike = 1.0,
    aux: ScalarLike | None = None,
) -> Array:
    r"""Gamma deviance contributions $2(r_i - \log(1 + r_i))$ where $r_i = y_i/\mu_i - 1$.

    Equivalent to $2\bigl((y_i - \mu_i)/\mu_i - \log(y_i/\mu_i)\bigr)$ but uses
    `log1p` for numerical stability when $y_i \approx \mu_i$.

    **Arguments:**

    - `y`: positive responses, shape `(n,)`.
    - `mu`: fitted means, shape `(n,)`.
    - `disp`: ignored for unscaled deviance.
    - `aux`: ignored.

    **Returns:**

    Non-negative deviance contributions, shape `(n,)`.
    """
    del disp, aux
    y_ = jnp.asarray(y)
    mu_ = jnp.asarray(mu)
    r_ = (y_ - mu_) / mu_
    return 2.0 * (r_ - jnp.log1p(r_))
```

**Testing** (append to `tests/family/test_families.py`):

```python
class TestCdfAllFamilies:
    def test_binomial_cdf_matches_scipy(self):
        f = Binomial()
        y = jnp.array([0.0, 1.0, 0.0, 1.0])
        mu = jnp.array([0.3, 0.7, 0.8, 0.2])
        result = f.cdf(y, mu)
        expected = jnp.array(scipy.stats.bernoulli.cdf(y, p=mu))
        assert jnp.allclose(result, expected, atol=1e-10)

    def test_gamma_cdf_matches_scipy(self):
        f = Gamma()
        y = jnp.array([1.0, 2.0, 3.0, 0.5])
        mu = jnp.array([2.0, 2.0, 2.0, 1.0])
        disp = 0.5
        result = f.cdf(y, mu, disp)
        k = 1.0 / disp
        theta = mu * disp
        expected = jnp.array(scipy.stats.gamma.cdf(y, a=k, scale=theta))
        assert jnp.allclose(result, expected, atol=1e-8)

    def test_nb_cdf_matches_scipy(self):
        f = NegativeBinomial()
        y = jnp.array([0.0, 2.0, 5.0, 10.0])
        mu = jnp.array([2.0, 3.0, 4.0, 8.0])
        alpha = 0.5
        r = 1.0 / alpha
        result = f.cdf(y, mu, aux=alpha)
        p_succ = mu / (r + mu)
        expected = jnp.array(scipy.stats.nbinom.cdf(y, n=r, p=1.0 - p_succ))
        assert jnp.allclose(result, expected, atol=1e-7)

    @pytest.mark.parametrize("FamilyCls", [Binomial, Gamma])
    def test_deviance_contribs_nonnegative(self, FamilyCls):
        f = FamilyCls()
        if isinstance(f, Gamma):
            y = jnp.array([1.0, 2.0, 3.0])
            mu = jnp.array([1.0, 2.0, 3.0])
        else:
            y = jnp.array([0.0, 1.0])
            mu = jnp.array([0.5, 0.5])
        result = f.deviance_contribs(y, mu)
        assert jnp.all(result >= 0)

    def test_nb_deviance_zero_y_finite(self):
        f = NegativeBinomial()
        y = jnp.array([0.0, 1.0, 3.0])
        mu = jnp.array([0.5, 1.5, 2.5])
        result = f.deviance_contribs(y, mu, aux=0.5)
        assert jnp.all(jnp.isfinite(result))

    @pytest.mark.parametrize("FamilyCls", [Gaussian, Poisson, Binomial, NegativeBinomial, Gamma])
    def test_cdf_shape_n(self, FamilyCls):
        n = 8
        f = FamilyCls()
        y = jnp.ones(n) * 1.0
        mu = jnp.ones(n) * 1.0
        kwargs = {"aux": 0.5} if FamilyCls is NegativeBinomial else {}
        if FamilyCls is Gamma:
            kwargs = {"disp": 1.0}
        result = f.cdf(y, mu, **kwargs)
        assert result.shape == (n,)

    @pytest.mark.parametrize("FamilyCls", [Gaussian, Poisson, Binomial, NegativeBinomial, Gamma])
    def test_deviance_contribs_shape_n(self, FamilyCls):
        n = 8
        f = FamilyCls()
        y = jnp.ones(n) * 1.0
        mu = jnp.ones(n) * 1.0
        kwargs = {"aux": 0.5} if FamilyCls is NegativeBinomial else {}
        result = f.deviance_contribs(y, mu, **kwargs)
        assert result.shape == (n,)
```

Note: `scipy` is already a test dependency (transitively via statsmodels). Add `import scipy.stats` at the top of the test file.

**Verification:**

Run: `pytest -p no:capture tests/family/`
Expected: All tests pass (including all 321 existing tests)

Run: `pytest -p no:capture tests`
Expected: All tests pass

**Commit:** `feat(family): implement cdf and deviance_contribs for Binomial, Gamma, and NegativeBinomial`
<!-- END_TASK_3 -->

<!-- END_SUBCOMPONENT_A -->
