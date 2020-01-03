

class my_property(object):

    error_msg = "'%s' must of be type %s but an object of type %s was given"

    def __init__(self, type_is=None, type_in=None, positive=False,
                 nonnegative=False):
        self.type_is = type_is
        self.type_in = type_in
        self.positive = positive
        self.nonnegative = nonnegative

    def __call__(self, func):

        @property
        def decorated_property(obj):
            pass

        @decorated_property.setter
        def decorated_property(obj, val):
            if self.type_is is not None and type(val) is not self.type_is:
                raise ValueError(my_property.error_msg % (func.__name__,
                                                       self.type_is.__name__,
                                                       val.__class__.__name__))
            elif self.type_in is not None and type(val) not in self.type_in:
                raise ValueError("BLABLA")
            elif self.positive and val <= 0:
                raise ValueError("Nonpositive")
            elif self.nonnegative and val < 0:
                raise ValueError("Not nonnegaive")
            else:
                print("func(obj, val)")
                func(obj, val)
                return obj

        return decorated_property


class C(object):

    def __init__(self):
        self._p = None

    @property
    def p(self):
        print('return self._p')
        return self._p

    @my_property(type_is=int, positive=True)
    def p(self, val):
        print("self._p = val")
        self._p = val


c = C()

# c.p = 'truc'
# print(c.p)

c.p = 32

print(c.p)
