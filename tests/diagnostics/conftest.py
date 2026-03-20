"""Shared fixture helpers for diagnostics tests."""

import numpy as np

import jax.numpy as jnp

import glmax


def fit_gaussian():
    X_raw = np.array([[1.0, -2.0], [1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y_raw = np.array(
        [
            1.8660254037844386,
            0.6339745962155614,
            2.0,
            1.6339745962155614,
            3.8660254037844384,
        ]
    )
    model = glmax.GLM(family=glmax.Gaussian())
    return glmax.fit(model, jnp.array(X_raw), jnp.array(y_raw)), X_raw, y_raw


def fit_poisson():
    X_raw = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 1.5]])
    y_raw = np.array([1.0, 2.0, 4.0, 7.0, 3.0])
    model = glmax.GLM(family=glmax.Poisson())
    return glmax.fit(model, jnp.array(X_raw), jnp.array(y_raw)), X_raw, y_raw


def fit_binomial():
    X_raw = np.ones((5, 1))
    y_raw = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    model = glmax.GLM(family=glmax.Binomial())
    return glmax.fit(model, jnp.array(X_raw), jnp.array(y_raw)), X_raw, y_raw


def fit_gamma():
    X_raw = np.array([[1.0, 0.5], [1.0, 1.0], [1.0, 1.5], [1.0, 2.0], [1.0, 2.5]])
    y_raw = np.array([0.8, 1.1, 1.7, 2.2, 2.9])
    model = glmax.GLM(family=glmax.Gamma())
    return glmax.fit(model, jnp.array(X_raw), jnp.array(y_raw)), X_raw, y_raw


def fit_negative_binomial():
    X_raw = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 1.5]])
    y_raw = np.array([0.0, 1.0, 2.0, 1.0, 4.0])
    model = glmax.GLM(family=glmax.NegativeBinomial())
    return glmax.fit(model, jnp.array(X_raw), jnp.array(y_raw)), X_raw, y_raw
