
This is ``linlearn``'s documentation
====================================

.. image:: https://travis-ci.org/linlearn/linlearn.svg?branch=master
   :target: https://travis-ci.org/linlearn/linlearn
.. image:: https://readthedocs.org/projects/linlearn/badge/?version=latest
   :target: https://linlearn.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://img.shields.io/pypi/pyversions/linlearn
   :alt: PyPI - Python Version
.. image:: https://img.shields.io/pypi/wheel/linlearn
   :alt: PyPI - Wheel
.. image:: https://img.shields.io/github/stars/linlearn/linlearn
   :alt: GitHub stars
   :target: https://github.com/linlearn/linlearn/stargazers
.. image:: https://img.shields.io/github/issues/linlearn/linlearn
   :alt: GitHub issues
   :target: https://github.com/linlearn/linlearn/issues
.. image:: https://img.shields.io/github/license/linlearn/linlearn
   :alt: GitHub license
   :target: https://github.com/linlearn/linlearn/blob/master/LICENSE
.. image:: https://coveralls.io/repos/github/linlearn/linlearn/badge.svg?branch=master
   :target: https://coveralls.io/github/linlearn/linlearn?branch=master

``linlearn`` simply stands for **linear learning**. It is a scikit-learn compatible python package for linear learning
with Python. It provides :

* Several strategies, including empirical risk minimization (which is the standard approach) and median-of-means for robust regression and classification

* Several loss functions easily accessible from a single class (``BinaryClassifier`` for binary classification and ``Regressor`` for regression)

* Several penalization functions, including standard L1, ridge and elastic-net, but also total-variation, slope, weighted L1, among many others

* All algorithms can use early stopping strategies during training

* Supports dense and sparse data formats, and includes fast solvers for large sparse datasets (using state-of-the-art stochastic optimization algorithms)

* It is accelerated thanks to numba, leading to a very concise, small, but fast library

Installation
------------

The easiest way to install linlearn is using pip

.. code-block:: bash

    pip install linlearn


But you can also use the latest development from github directly with

.. code-block:: bash

    pip install git+https://github.com/linlearn/linlearn.git

References
----------

Usage
-----

``linlearn`` follows the scikit-learn API: you call fit instead of use ``predict_proba``
or ``predict`` whenever you need predictions.

.. code-block:: python

   from linlearn import BinaryClassifier

   clf = BinaryClassifier()
   clf.fit(X_train, y_train)
   y_pred = clf.predict_proba(X_test)[:, 1]


Where to go from here?
----------------------

.. toctree::
   :maxdepth: 2
   :hidden:
