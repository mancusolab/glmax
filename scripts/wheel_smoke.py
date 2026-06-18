# pattern: Imperative Shell

import jax.numpy as jnp

import glmax


def main() -> None:
    print(f"imported glmax from {glmax.__file__}")

    X = jnp.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, 1.5]])
    y = jnp.array([1.0, 2.0, 2.0, 4.0])

    fitted = glmax.fit(glmax.Poisson(), X, y)
    pred = glmax.predict(fitted.family, fitted.params, X)
    result = glmax.infer(fitted)
    diag = glmax.check(fitted)

    if pred.shape != y.shape:
        raise AssertionError(f"prediction shape mismatch: expected {y.shape}, got {pred.shape}")
    if result.params.beta.shape != (X.shape[1],):
        raise AssertionError("inference parameter shape mismatch")
    if diag.deviance.shape != ():
        raise AssertionError("goodness-of-fit statistic must be scalar")

    print("installed glmax smoke passed")


if __name__ == "__main__":
    main()
