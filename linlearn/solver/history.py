# License: BSD 3 clause
from collections import defaultdict

import warnings


# We want to import the tqdm.autonotebook but don't want to see the warning...
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import trange


class History(object):
    """

    """
    def __init__(self, title, max_iter, verbose):
        self.max_iter = max_iter
        self.verbose = verbose
        self.keys = None
        self.values = defaultdict(list)
        self.title = title
        self.n_updates = 0
        # TODO: List all possible keys
        print_style = defaultdict(lambda: '%.2e')
        print_style.update(**{
            'n_iter': '%d',
            'epoch': '%d',
            'n_inner_prod"': '%d',
            'spars': '%d',
            'rank': '%d',
            'tol': '%.2f'
        })
        self.print_style = print_style

        # The progress bar using tqdm
        if self.verbose:
            bar_format = "{desc} : {percentage:2.0f}% {bar} epoch: {n_fmt} " \
                         "/ {total_fmt} , elapsed: {elapsed_s:3.1f}s {postfix}"
            self.bar = trange(max_iter, desc=title, unit=' epoch ',
                              leave=True, bar_format=bar_format)

    def update(self, update_bar=True, **kwargs):
        # Total number of calls to update must be smaller than max_iter + 1
        if self.max_iter >= self.n_updates:
            self.n_updates += 1
        else:
            raise ValueError("Already %d updates while max_iter=%d" %
                             (self.n_updates, self.max_iter))

        # The first time update is called it establishes the list of keys to be
        # given to the history. Following calls to update must provide the
        # exact same keys
        if self.keys is None:
            # OK since order is preserved in kwargs since Python 3.6
            self.keys = list(kwargs.keys())
        else:
            k1, k2 = set(self.keys), set(kwargs.keys())
            if k1 != k2:
                raise ValueError("'update' excepted the following keys: %s "
                                 "must received %s instead" % (k1, k2))

        values = self.values
        print_style = self.print_style

        # Update the history
        for key, val in kwargs.items():
            values[key].append(val)

        # If required, update the tqdm bar
        if self.verbose and update_bar:
            postfix = ' , '.join([
                key + ': ' + str(print_style[key] % val)
                for key, val in kwargs.items()
            ])
            self.bar.set_postfix_str(postfix)
            self.bar.update(1)

    def close_bar(self):
        self.bar.close()

    def clear(self):
        """Reset history values"""
        self.values = defaultdict(list)
        self.keys = None
        self.n_updates = 0

    def print(self):
        keys = self.keys
        values = self.values
        print("keys: ", keys)
        min_width = 9
        line = ' | '.join([
                key.center(min_width) for key in keys
            ])
        names = [key.center(min_width) for key in keys]

        col_widths = list(map(len, names))
        print(line)

        print_style = self.print_style
        n_lines = len(list(values.values())[0])
        for i_line in range(n_lines):
            line = ' | '.join([
                str(print_style[key] % values[key][i_line])
                    .rjust(col_widths[i])
                for i, key in enumerate(keys)
            ])
            print(line)
