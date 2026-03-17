# Equation To Code Map

## Context
- Plan slug: `params-disp-aux`
- Generated date: `2026-03-17`

| Equation ID | Equation (LaTeX or text) | Intended Computation | Target Module/Function | Test ID | Status |
| --- | --- | --- | --- | --- | --- |
| EQ-1 | `eta = X beta + offset` | Linear predictor used by fit and predict. | `src/glmax/_fit/irls.py`, `src/glmax/_fit/fit.py` | `params-disp-aux.AC3.1` | planned |
| EQ-2 | `mu = g^{-1}(eta)` | Mean response from the active link. | `src/glmax/glm.py::mean`, family link implementations | `params-disp-aux.AC3.3` | planned |
| EQ-3 | `Var(Y | mu) = phi V(mu)` | EDM dispersion semantics for families that estimate or carry `phi`. | `src/glmax/family/dist.py`, `src/glmax/glm.py`, inference estimators | `params-disp-aux.AC2.1`, `params-disp-aux.AC3.2` | planned |
| EQ-4 | `Var(Y | mu) = mu + alpha mu^2` | Negative Binomial variance with auxiliary `alpha`, not GLM `phi`. | `src/glmax/family/dist.py::NegativeBinomial.variance` | `params-disp-aux.AC2.3` | planned |
| EQ-5 | `Cov(beta_hat) = phi I(beta_hat)^{-1}` | Fisher-information covariance scaling for downstream inference. | `src/glmax/_infer/stderr.py::FisherInfoError` | `params-disp-aux.AC3.2` | planned |

## Checks
- [x] Objective sign and optimization direction are correct.
- [x] Update rules map to concrete computation steps.
- [x] Every mapped equation has a corresponding test target.
