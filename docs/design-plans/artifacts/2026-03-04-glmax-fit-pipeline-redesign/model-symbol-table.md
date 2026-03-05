# Model Symbol Table

## Context
- Plan slug: `glmax-fit-pipeline-redesign`
- Generated date: `2026-03-04`

| Symbol | Meaning | Domain/Support | Shape/Type | Defined In Source | Notes |
| --- | --- | --- | --- | --- | --- |
| $X$ | Design matrix | Real-valued covariates | `ArrayLike`, shape `(n, p)` | `src/glmax/fit.py` | Boundary-validated as numeric and finite |
| $y$ | Response vector | Family-dependent response support | `ArrayLike`, shape `(n,)` | `src/glmax/fit.py` | Boundary-validated as numeric and finite |
| $\\eta$ | Linear predictor | Real line | `Array`, shape `(n,)` | `src/glmax/infer/optimize.py` | Computed as $X\\beta + \\text{offset}$ |
| $\\mu$ | Mean response | Family-dependent mean support | `Array`, shape `(n,)` | `src/glmax/family/dist.py` | Computed by inverse link $g^{-1}(\\eta)$ |
| $\\beta$ | Regression coefficients | Real vector | `Array`, shape `(p,)` | `src/glmax/infer/optimize.py` | Solved each IRLS iteration |
| $\\alpha$ | Dispersion parameter | Positive reals for NB, zero otherwise | scalar `Array` | `src/glmax/family/dist.py` | Updated in NB-specific path |
| $W$ | IRLS weight vector | Positive reals | `Array`, shape `(n,)` | `src/glmax/family/dist.py` | From variance, scale, and link derivative |

## Checks
- [ ] No undefined symbols.
- [ ] No conflicting symbol reuse.
- [ ] Support/domain constraints are explicit.
