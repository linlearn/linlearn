# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module contains all the solvers available in ``linlearn``
"""

from .cgd import CGD
from .gd import GD, batch_GD
from .md import MD
from .da import DA
from .sgd import SGD
from .saga import SAGA
from .svrg import SVRG
from .history import History, plot_history
