[![Documentation-webpage](https://img.shields.io/badge/Docs-Available-brightgreen)](https://mancusolab.github.io/glmax/)
[![PyPI-Server](https://img.shields.io/pypi/v/glmax.svg)](https://pypi.org/project/glmax/)
[![Github](https://img.shields.io/github/stars/mancusolab/glmax?style=social)](https://github.com/mancusolab/glmax)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project generated with Hatch](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

# GLMax
`glmax` is is a Python library to TBD..

  [**Installation**](#installation)
  | [**Example**](#get-started-with-example)
  | [**Notes**](#notes)
  | [**Version**](#version-history)
  | [**Support**](#support)
  | [**Other Software**](#other-software)

------------------

## Installation

Users can download the latest repository and then use `pip`:

``` bash
git clone https://github.com/mancusolab/glmax.git
cd glmax
pip install .
```

## Get Started with Example

`glmax.fit` is the canonical fit entrypoint. `GLM.fit` remains available as a
compatibility wrapper and routes through the same normalization and fitter
orchestration path.

```python
import jax.numpy as jnp
import glmax

X = jnp.array([[1.0, 0.2], [1.0, -0.3], [1.0, 1.1], [1.0, -0.9]])
y = jnp.array([1.0, 0.0, 2.0, 1.0])

state = glmax.fit(X, y, family=glmax.Poisson(), solver=glmax.CholeskySolver())
compat_state = glmax.GLM(family=glmax.Poisson(), solver=glmax.CholeskySolver()).fit(X, y)
```

## Notes

- Fit-boundary failures are deterministic and shared by canonical and wrapper entrypoints:
  - `TypeError` for non-numeric `X`, `y`, `offset_eta`, `init`, or `alpha_init`
  - `ValueError` for rank/shape mismatches
  - `ValueError` for non-finite boundary inputs (NaN/Inf)

-   `glmax` uses [JAX](https://github.com/google/jax) with [Just In
    Time](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)
    compilation to achieve high-speed computation. However, there are
    some [issues](https://github.com/google/jax/issues/5501) for JAX
    with Mac M1 chip. To solve this, users need to initiate conda using
    [miniforge](https://github.com/conda-forge/miniforge), and then
    install `glmax` using `pip` in the desired environment.

## Version History

TBD

## Support

Please report any bugs or feature requests in the [Issue
Tracker](https://github.com/mancusolab/glmax/issues). If users have
any questions or comments, please contact Zixuan Zhang (<zzhang39@usc.edu>)
and Nicholas Mancuso (<nmancuso@usc.edu>).

## Other Software

Feel free to use other software developed by [Mancuso
Lab](https://www.mancusolab.com/):

-   [SuShiE](https://github.com/mancusolab/sushie): a Bayesian
    fine-mapping framework for molecular QTL data across multiple
    ancestries.
-   [perturbVI](https://github.com/mancusolab/perturbVI): a Bayesian
    framework for quantifying perturbation effects and regulatory modules
    in large-scale CRISPR screens.
-   [MA-FOCUS](https://github.com/mancusolab/ma-focus): a Bayesian
    fine-mapping framework using
    [TWAS](https://www.nature.com/articles/ng.3506) statistics across
    multiple ancestries to identify the causal genes for complex traits.
-   [SuSiE-PCA](https://github.com/mancusolab/susiepca): a scalable
    Bayesian variable selection technique for sparse principal component
    analysis
-   [twas_sim](https://github.com/mancusolab/twas_sim): a Python
    software to simulate [TWAS](https://www.nature.com/articles/ng.3506)
    statistics.
-   [FactorGo](https://github.com/mancusolab/factorgo): a scalable
    variational factor analysis model that learns pleiotropic factors
    from GWAS summary statistics.
-   [HAMSTA](https://github.com/tszfungc/hamsta): a Python software to
    estimate heritability explained by local ancestry data from
    admixture mapping summary statistics.

------------------------------------------------------------------------

`glmax` is distributed under the terms of the
[MIT](https://spdx.org/licenses/MIT.html) license.


------------------------------------------------------------------------

This project has been set up using Hatch. For details and usage
information on Hatch see <https://github.com/pypa/hatch>.
