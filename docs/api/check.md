# `check`

`glmax.check(...)` computes diagnostics from a fitted noun without refitting.
The high-level philosophy is that diagnostics are explicit strategy objects:
`check` applies one concrete diagnostic at a time and returns the typed result
for that diagnostic, which keeps the workflow easy to compose and type-check.

::: glmax.check

---

## Diagnostic Contract

`AbstractDiagnostic` defines the strategy interface behind `check`.

::: glmax.AbstractDiagnostic

## Built-In Diagnostic Strategies

::: glmax.PearsonResidual

---

::: glmax.DevianceResidual

---

::: glmax.QuantileResidual

---

::: glmax.GoodnessOfFit

---

::: glmax.Influence

## Diagnostic Result Nouns

Some diagnostics return arrays directly, while others return explicit result
nouns.

### `GofStats`

::: glmax.GofStats

### `InfluenceStats`

::: glmax.InfluenceStats
