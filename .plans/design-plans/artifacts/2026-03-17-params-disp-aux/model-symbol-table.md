# Model Symbol Table

## Context
- Plan slug: `params-disp-aux`
- Generated date: `2026-03-17`

| Symbol | Meaning | Domain/Support | Shape/Type | Defined In Source | Notes |
| --- | --- | --- | --- | --- | --- |
| `beta` | Regression coefficient vector | finite reals | rank-1 inexact array `(p,)` | `src/glmax/_fit/types.py` | Public coefficient carrier in `Params`. |
| `phi` / `disp` | GLM/EDM dispersion parameter | positive real scalar or canonical fixed `1.0` | inexact scalar | `src/glmax/_fit/types.py`, `src/glmax/family/dist.py` | Stored in `Params.disp`; inference scaling reads this value directly. |
| `aux` | Family-specific auxiliary parameter | `None` or finite scalar; family-specific support | optional inexact scalar | planned contract in `src/glmax/_fit/types.py` and `src/glmax/family/dist.py` | NB uses `aux = alpha`; unsupported families require `aux is None`. |
| `alpha` | Negative Binomial overdispersion / ancillary parameter | positive real scalar | inexact scalar | `src/glmax/family/dist.py` | Moves from current overloaded `disp` meaning into `aux`. |
| `eta` | Linear predictor | real vector | rank-1 array `(n,)` | `src/glmax/glm.py`, `src/glmax/_fit/irls.py` | `eta = X beta + offset`. |
| `mu` | Mean response | family support | rank-1 array `(n,)` | `src/glmax/glm.py`, `src/glmax/family/dist.py` | `mu = g^{-1}(eta)`. |
| `glm_wt` | GLM working weights | non-negative reals | rank-1 array `(n,)` | `src/glmax/_fit/types.py`, `src/glmax/family/dist.py` | Depends on variance and link derivative; semantics must follow `disp`/`aux` split. |

## Checks
- [x] No undefined symbols.
- [x] No conflicting symbol reuse.
- [x] Support/domain constraints are explicit.
