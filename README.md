
[![Build Status](https://travis-ci.com/linlearn/linlearn.svg?branch=master)](https://travis-ci.com/linlearn/linlearn)
[![Documentation Status](https://readthedocs.org/projects/linlearn/badge/?version=latest)](https://linlearn.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/linlearn)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/linlearn)
[![GitHub stars](https://img.shields.io/github/stars/linlearn/linlearn)](https://github.com/linlearn/linlearn/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/linlearn/linlearn)](https://github.com/linlearn/linlearn/issues)
[![GitHub license](https://img.shields.io/github/license/linlearn/linlearn)](https://github.com/linlearn/linlearn/blob/master/LICENSE)
[![Coverage Status](https://coveralls.io/repos/github/linlearn/linlearn/badge.svg?branch=master)](https://coveralls.io/github/linlearn/linlearn?branch=master)

# linlearn: linear methods in Python

LinLearn is scikit-learn compatible python package for machine learning with linear methods. 
It includes in particular alternative "strategies" for robust training, including median-of-means for classification and regression.

[Documentation](https://linlearn.readthedocs.io) | [Reproduce experiments](https://linlearn.readthedocs.io/en/latest/linlearn.html) |

LinLearn simply stands for linear learning. It is a small scikit-learn compatible python package for **linear learning** 
with Python. It provides :

- Several strategies, including empirical risk minimization (which is the standard approach), 
median-of-means for robust regression and classification
- Several loss functions easily accessible from a single class (`BinaryClassifier` for classification and `Regressor` for regression)
- Several penalization functions, including standard L1, ridge and elastic-net, but also total-variation, slope, weighted L1, among many others
- All algorithms can use early stopping strategies during training
- Supports dense and sparse format, and includes fast solvers for large sparse datasets (using state-of-the-art stochastic optimization algorithms) 
- It is accelerated thanks to numba, leading to a very concise, small, but very fast library
  
## Installation

The easiest way to install linlearn is using pip

    pip install linlearn

But you can also use the latest development from github directly with

    pip install git+https://github.com/linlearn/linlearn.git

## References
