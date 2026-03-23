# Fitting Strategies And Solvers

Most users should stay with `glmax.fit(family, X, y)` and the default IRLS
strategy. This page documents the lower-level strategy and solver objects for
users who need to control the numerical fitting path explicitly.

## Fitter Strategies

??? abstract "`glmax.AbstractFitter`"

    ::: glmax.AbstractFitter
        options:
            members:
                - fit

::: glmax.IRLSFitter

## Linear Solvers

Weighted least-squares steps are delegated to linear solvers. These remain
advanced tools under rather than primary package-root workflow
objects.

??? abstract "`glmax.AbstractLinearSolver`"

    ::: glmax.AbstractLinearSolver


::: glmax.CholeskySolver

---

::: glmax.QRSolver

---

::: glmax.CGSolver

