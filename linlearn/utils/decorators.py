

class val_type(object):

    error_msg = "'%s' must of be type %s but an object of type %s was given"

    def __init__(self, type):
        self.type = type

    def __call__(self, func):
        def decorated_func(obj, x):
            if type(x) is self.type:
                return func(obj, x)
            else:
                raise ValueError(val_type.error_msg % (func.__name__,
                                                       self.type.__name__,
                                                       x.__class__.__name__))
        return decorated_func


class arg_nonnegative(object):

    error_msg = "'%s' must of be type %s but an object of type %s was given"

    def __init__(self, type):
        self.type = type

    def __call__(self, func):
        def decorated_func(obj, x):
            if type(x) is self.type:
                return func(obj, x)
            else:
                raise ValueError(arg_nonnegative.error_msg % (func.__name__,
                                                     self.type.__name__,
                                                     x.__class__.__name__))
        return decorated_func
