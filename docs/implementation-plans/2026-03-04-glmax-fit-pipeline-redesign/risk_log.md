# GLMAX Fit Pipeline Redesign Residual Risk Log

## Status

This log captures residual risks after implementation through Phase 6 Task 3.

## Risks

| ID | Risk | Severity | Mitigation | Owner | Status | Final Disposition |
| --- | --- | --- | --- | --- | --- |
| R1 | `lineax.NormalCG` deprecation warning may become a hard break in a future dependency upgrade. | Medium | Track Lineax release notes; migrate `CGSolver` to `lx.Normal(lx.CG(...))` before the next dependency refresh cycle. | GLMAX maintainers | Open | Accepted with follow-up action in dependency-maintenance cycle. |
| R2 | `GLM.fit` compatibility shim could remain in code longer than intended if deprecation checkpoints are not actively enforced. | Medium | Keep deprecation checkpoints in docs and handoff gate, and require parity + boundary tests to pass before any shim-change decision. | GLMAX maintainers | Open | Accepted with explicit exit-gate enforcement in `handoff.md`. |
| R3 | Documentation claims could drift from behavior over time as fit and inference surfaces evolve. | Low | Keep docs-linked regression assertions in `tests/test_fit_api.py`, `tests/test_fitters.py`, and `tests/test_glm.py`. | GLMAX maintainers | Mitigated | Closed by docs-linked regression assertions currently passing. |
| R4 | Numerical parity tolerances may regress under future JAX/Lineax updates even if APIs stay stable. | Medium | Run full `pytest -p no:capture tests` matrix in CI for every dependency update and keep parity tests across Gaussian/Poisson/Binomial/Negative Binomial families. | GLMAX maintainers | Open | Accepted with CI parity matrix as ongoing monitoring control. |

## Notes

- No high-severity residual risks were identified during this redesign.
- Final disposition review completed on 2026-03-05.
