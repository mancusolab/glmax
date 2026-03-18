# Params Disp Aux Implementation Plan

This directory contains the execution artifacts for implementing the
`params-disp-aux` design.

## Design Source

- Design plan:
  `/Users/nicholas/Projects/glmax/.worktrees/params-disp-aux/.plans/design-plans/2026-03-17-params-disp-aux.md`

## Acceptance-Criteria Coverage

- `params-disp-aux.AC1.*`: parameter-carrier contract and GLM canonicalization
- `params-disp-aux.AC2.*`: family-level `disp` versus `aux` semantics
- `params-disp-aux.AC3.*`: fit, predict, and infer plumbing for canonical
  `Params(beta, disp, aux)`
- `params-disp-aux.AC4.*`: documentation, contributor guidance, and regression
  terminology alignment

## Phase Index

- `phase_01.md`
  - Adds `Params(beta, disp, aux)` and GLM-level parameter canonicalization.
- `phase_02.md`
  - Splits family dispersion and auxiliary semantics and routes them through
    `GLM` methods.
- `phase_03.md`
  - Threads canonical parameters through fitting, prediction, and inference.
- `phase_04.md`
  - Aligns public docs, contributor guidance, and contract tests with the new
    terminology.

## Verification Artifacts

- `test-requirements.md`
  - Maps automated and human verification requirements to
    `params-disp-aux.AC1.*` through `params-disp-aux.AC4.*`.
