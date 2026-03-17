# Solver Feasibility Matrix

## Context
- Plan slug: `params-disp-aux`
- Generated date: `2026-03-17`

| Candidate | Problem Form Fit (root/least-squares/minimize) | AD Compatibility | Constraint Handling | Status/Error Mapping | Feasible (yes/no) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Optimistix | Not required for current IRLS refactor | Compatible in principle | Would require new constrained update path for `aux` | Adds migration risk unrelated to terminology cleanup | no | Keep out of scope for this design. |
| Lineax | Existing weighted least-squares solves inside IRLS | Already in use | Existing constraints remain sufficient; only parameter plumbing changes | Existing linear-solve failures remain the operative behavior | yes | Preferred because solver behavior is already covered by current fit tests. |
| Custom Solver | Possible but unnecessary | Would need fresh validation | Could encode positivity constraints for `aux`, but at high cost | New error semantics would complicate regression review | no | Not justified for a semantic contract refactor. |

## Decision
- Preferred solver path: retain the current `lineax`-backed IRLS fitter.
- Reason: no solver change is needed to separate `disp` from `aux`.
- Benchmark or validation requirement before implementation: none beyond the existing fit/infer regression suite and family-specific convergence checks.
