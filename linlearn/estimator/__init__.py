# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module contains all the estimators available in ``linlearn``
"""

from .erm import ERM, StateERM
from .mom import MOM, StateMOM
from .ch import CH, StateCH
from .llm import LLM, StateLLM
from .gmom import GMOM, StateGMOM
from .tmean import TMean, TMean_variant, StateTMean
from .hg import HG, StateHG
from .dkk import DKK, StateDKK