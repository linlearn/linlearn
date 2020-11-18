
Classification
==============

For now, ``linlearn`` provides mainly the scikit-learn compatible :obj:`BinaryClassifier` class for binary classification.
Its usage follows the ``scikit-learn`` API, namely a ``fit``, ``predict_proba``
and ``predict`` methods to respectively fit, predict class probabilities and labels.
The :obj:`BinaryClassifier` with default parameters is created using

.. code-block:: python

    from linlearn import BinaryClassifier

    amf = AMFClassifier()

.. autosummary::
   :toctree: generated/

   linlearn.BinaryClassifier
