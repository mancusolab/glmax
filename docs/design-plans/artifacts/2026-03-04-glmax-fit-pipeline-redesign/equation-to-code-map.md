# Equation To Code Map

## Context
- Plan slug: `glmax-fit-pipeline-redesign`
- Generated date: `2026-03-04`

| Equation ID | Equation (LaTeX or text) | Intended Computation | Target Module/Function | Test ID | Status |
| --- | --- | --- | --- | --- | --- |
| EQ-1 | $\\eta = X\\beta + \\text{offset}$ | Compute linear predictor from coefficients and offset | `infer` fitter loop (`IRLSFitter` target; currently `infer/optimize.py`) | `test_gx_fit_returns_glmstate_for_gaussian` | mapped |
| EQ-2 | $\\mu = g^{-1}(\\eta)$ | Map linear predictor to mean via inverse link | `family.glink.inverse` | `test_wrapper_and_direct_entrypoints_have_output_parity` | mapped |
| EQ-3 | $w_i = [\\phi V(\\mu_i) (g'(\\mu_i))^2]^{-1}$ | Compute IRLS working weights | `ExponentialFamily.calc_weight` | `test_poisson`/`test_logit` regression checks | mapped |
| EQ-4 | $r = \\eta + g'(\\mu)(y-\\mu) - \\text{offset}$ | Compute working response for weighted solve | `infer` fitter loop (`IRLSFitter` target; currently `infer/optimize.py`) | `test_gx_fit_matches_glm_fit_convergence_metadata` | mapped |
| EQ-5 | $\\beta \\leftarrow \\arg\\min_{\\beta} \\|W^{1/2}(X\\beta-r)\\|_2^2$ | Solve weighted least-squares update | `AbstractLinearSolver.__call__` and concrete solvers | `test_poisson`/`test_normal`/`test_logit` | mapped |
| EQ-6 | $z = \\beta / \\mathrm{se}(\\beta)$ and two-sided p-value | Post-fit significance testing | `WaldTest`/fit pipeline | `test_wrapper_and_direct_hypothesis_method_parity` | mapped |

## Checks
- [ ] Objective sign and optimization direction are correct.
- [ ] Update rules map to concrete computation steps.
- [ ] Every mapped equation has a corresponding test target.
