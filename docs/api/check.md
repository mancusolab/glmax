# Model Diagnostics

`glmax.check(...)` computes diagnostics from a fitted noun without refitting.
The high-level philosophy is that diagnostics are explicit strategy objects:
`check` applies one concrete diagnostic at a time and returns the typed result
for that diagnostic, which keeps the workflow easy to compose and type-check.

::: glmax.check

---

## Diagnostics

`AbstractDiagnostic` defines the strategy interface behind `check`.

??? abstract "glmax.AbstractDiagnostic"

    ::: glmax.AbstractDiagnostic


::: glmax.PearsonResidual

---

::: glmax.DevianceResidual

---

::: glmax.QuantileResidual

---

::: glmax.GoodnessOfFit

---

::: glmax.Influence

## Diagnostic results

::: glmax.GofStats

::: glmax.InfluenceStats
