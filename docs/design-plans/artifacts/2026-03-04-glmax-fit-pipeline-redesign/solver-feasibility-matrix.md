# Solver Feasibility Matrix

## Context
- Plan slug: `glmax-fit-pipeline-redesign`
- Generated date: `2026-03-04`

| Candidate | Problem Form Fit (root/least-squares/minimize) | AD Compatibility | Constraint Handling | Status/Error Mapping | Feasible (yes/no) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Optimistix | Strong for deterministic nonlinear/root routines; less direct for current linear WLS step | Good | N/A for current linear solve boundary | Would require introducing new nonlinear contract not currently exposed | no (for this redesign scope) | Keep out of scope; not needed for structural refactor |
| Lineax | Direct fit for weighted linear solves in IRLS iteration | Good for current usage | Existing solver types already model structure assumptions | Existing exception-first behavior; could be extended to explicit result channel later | yes | Current solver backend already validated and integrated |
| Custom Solver | Possible but unnecessary for current objective | Variable and maintenance-heavy | Must be implemented and maintained manually | Would require new failure contract and extensive tests | no | Adds risk without solving current architecture problem |

## Decision
- Preferred solver path: Keep Lineax-based solver strategies (`QRSolver`, `CGSolver`, `CholeskySolver`) during consolidation.
- Reason: Refactor goal is module/API simplification with behavior parity; solver replacement is unnecessary scope expansion.
- Benchmark or validation requirement before implementation: Preserve existing solver regression parity vs StatsModels and current test baselines.
