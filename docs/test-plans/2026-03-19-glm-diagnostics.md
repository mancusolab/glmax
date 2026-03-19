# GLM Diagnostics Human Test Plan

- Run a package-root import smoke test: `python -c "import glmax; print(glmax.DEFAULT_DIAGNOSTICS)"`.
- Fit a small Gaussian and Poisson model, then call `glmax.check(fitted)` and confirm the default residual output is finite. Also confirm `jax.tree_util.tree_map(..., is_leaf=...)` over `DEFAULT_DIAGNOSTICS` produces the expected tuple of concrete results.
- Fit Gamma and Negative Binomial models and manually call `QuantileResidual().diagnose(fitted)` to confirm finite residuals on the non-Gaussian paths.
- Confirm the BIC convention in `tests/diagnostics/test_gof.py` matches the log-likelihood-based formula documented in `GofStats`.
