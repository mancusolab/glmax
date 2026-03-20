# Model Inference

`glmax.infer(...)` computes inferential summaries from a fitted noun without
refitting the model. The high-level philosophy is that fitting and inference
are separate verbs: a fit produces parameter estimates and artifacts, and
`infer` turns those artifacts into standard errors, test statistics, and
p-values through explicit inference strategies.

::: glmax.infer

---

::: glmax.InferenceResult

