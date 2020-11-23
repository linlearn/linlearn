import numbers
import numpy as np


def f(*, truc, machin, **kwargs):

    print(truc, machin)
    print(kwargs)


f(truc=1, machin=2, chose="bidule")
