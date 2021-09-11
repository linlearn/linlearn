# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module contains all the classes for solvers (first-order optimization
algorithms) used in `linlearn`.
"""

from ._base import Solver
from ._cgd import CGD
from ._gd import GD
from ._sgd import SGD
from ._svrg import SVRG
from ._saga import SAGA
from ._history import History, plot_history
