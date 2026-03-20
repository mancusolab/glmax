# Model Prediction

`glmax.predict(...)` applies a model specification and fitted parameters to
data and returns mean predictions. The high-level philosophy is that prediction
should stay explicit about both the model and the parameter carrier rather than
hiding state inside a fitted object method.

::: glmax.predict

