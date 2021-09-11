"""
This module implements the `History` class, used in all solvers to keep track of the
iterations history.

References
"""

# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

from collections import defaultdict
import warnings
import numpy as np
import matplotlib.pyplot as plt

# We want to import the tqdm.autonotebook but don't want to see the warning...
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import trange


class Record(object):
    def __init__(self, shape, capacity):
        self.record = (
            np.zeros(capacity)
            if shape == 1
            else np.zeros(tuple([capacity] + list(shape)))
        )
        self.cursor = 0

    def update(self, value):
        self.record[self.cursor] = value
        self.cursor += 1

    def __len__(self):
        return self.record.shape[0]


class History(object):
    """

    """

    def __init__(self, title, max_iter, verbose, trackers=None):
        self.max_iter = max_iter
        self.verbose = verbose
        self.keys = None
        self.values = defaultdict(list)
        self.title = title
        self.trackers = [tracker[0] for tracker in trackers] if trackers else None
        self.records = (
            [Record(tracker[1], max_iter + 1) for tracker in trackers]
            if trackers
            else None
        )
        self.n_updates = 0
        # TODO: List all possible keys
        print_style = defaultdict(lambda: "%.2e")
        print_style.update(
            **{
                "n_iter": "%d",
                "epoch": "%d",
                'n_inner_prod"': "%d",
                "spars": "%d",
                "rank": "%d",
                "tol": "%.2e",
            }
        )
        self.print_style = print_style

        # The progress bar using tqdm
        if self.verbose:
            bar_format = (
                "{desc} : {percentage:2.0f}% {bar} epoch: {n_fmt} "
                "/ {total_fmt} , elapsed: {elapsed_s:3.1f}s {postfix}"
            )
            self.bar = trange(
                max_iter, desc=title, unit=" epoch ", leave=True, bar_format=bar_format
            )
        else:
            self.bar = None

    def update(self, current_iterate, update_bar=True, **kwargs):
        # Total number of calls to update must be smaller than max_iter + 1
        if self.max_iter >= self.n_updates:
            self.n_updates += 1
        else:
            raise ValueError(
                "Already %d updates while max_iter=%d" % (self.n_updates, self.max_iter)
            )

        # The first time update is called it establishes the list of keys to be
        # given to the history. Following calls to update must provide the
        # exact same keys
        if self.keys is None:
            # OK since order is preserved in kwargs since Python 3.6
            self.keys = list(kwargs.keys())
        else:
            k1, k2 = set(self.keys), set(kwargs.keys())
            if k1 != k2:
                raise ValueError(
                    "'update' excepted the following keys: %s "
                    "must received %s instead" % (k1, k2)
                )

        values = self.values
        print_style = self.print_style

        # Update the history
        for key, val in kwargs.items():
            values[key].append(val)

        # If required, update the tqdm bar
        if self.verbose and update_bar:
            postfix = " , ".join(
                [
                    key + ": " + str(print_style[key] % val)
                    for key, val in kwargs.items()
                ]
            )
            self.bar.set_postfix_str(postfix)
            self.bar.update(1)

        if self.trackers:
            for ind_tracker, tracker in enumerate(self.trackers):
                self.records[ind_tracker].update(tracker(current_iterate))

    def close_bar(self):
        if self.bar is not None:
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
        line = " | ".join([key.center(min_width) for key in keys])
        names = [key.center(min_width) for key in keys]

        col_widths = list(map(len, names))
        print(line)

        print_style = self.print_style
        n_lines = len(list(values.values())[0])
        for i_line in range(n_lines):
            line = " | ".join(
                [
                    str(print_style[key] % values[key][i_line]).rjust(col_widths[i])
                    for i, key in enumerate(keys)
                ]
            )
            print(line)


# Matplotlib colors of tab cmap (previously called Vega)
# It has been re-ordered so that light colors apperas at the end
tab20_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]


def get_plot_color(number):
    """Return color for a line number.
    Color are extracted from tab20 colormap which is an extension of
    matplotlib 2.x CN colors. 20 colors are available.

    Parameters
    ----------
    number : `int`
        Number of the color to pick

    Returns
    -------
    color : `str`
        Color in hexadecimal format
    """
    return tab20_colors[number % len(tab20_colors)]


# TODO: use n_iter instead or pass, epoch or whatnot


def extract_history(solvers, x, y, given_labels):
    x_arrays = []
    y_arrays = []
    labels = []
    for i, solver in enumerate(solvers):
        if hasattr(solver, "history_"):
            history = solver.history_
            if not isinstance(history, History):
                raise ValueError("Not an history !")
        else:
            raise ValueError("Object %s has no history" % solver.__class__.__name__)

        # If label was not given we override it
        if given_labels is not None and i < len(given_labels):
            # Better test
            label = given_labels[i]
        else:
            label = solver.__class__.__name__

        if x not in history.values.keys():
            raise ValueError("%s has no history for %s" % (label, x))
        elif y not in history.values.keys():
            raise ValueError("%s has no history for %s" % (label, y))
        else:
            x_arrays.append(np.array(history.values[x]))
            y_arrays.append(np.array(history.values[y]))
            labels.append(label)
    return x_arrays, y_arrays, labels


# TODO: defaults for x and


def plot_history(
    solvers,
    x,
    y,
    labels=None,
    show=True,
    log_scale: bool = False,
    dist_min: bool = False,
    rendering: str = "matplotlib",
    ax=None,
):
    """Plot the history of convergence of learners or solvers.

    It is used to compare easily their convergence performance.

    Parameters
    ----------
    solvers : `list` of `object`
        A list of solvers or learners with an history attribute to plot

    x : `str`
        The element of history to use as x-axis

    y : `str`, default='obj'
        The element of history to use as y-axis

    labels : `list` of `str`, default=None
        Label of each solver_old in the legend. If set to None then the class
        name of each solver_old will be used.

    show : `bool`, default=`True`
        if `True`, show the plot. Otherwise an explicit call to the show
        function is necessary. Useful when superposing several plots.

    log_scale : `bool`, default=`False`
        If `True`, then y-axis is on a log-scale

    dist_min : `bool`, default=`False`
        If `True`, plot the difference between `y` of each solver_old and the
        minimal `y` of all solvers. This is useful when comparing solvers on
        a logarithmic scale, to illustrate linear convergence of algorithms

    rendering : {'matplotlib', 'bokeh'}, default='matplotlib'
        Rendering library. 'bokeh' might fail if the module is not installed.

    ax : `list` of `matplotlib.axes`, default=None
        If not None, the figure will be plot on this axis and show will be
        set to False. Used only with matplotlib
    """
    x_arrays, y_arrays, labels = extract_history(solvers, x, y, labels)

    if dist_min:
        min_y = np.min(np.hstack(y_arrays))
        y_arrays = [y_array - min_y for y_array in y_arrays]

    min_x, max_x = np.min(np.hstack(x_arrays)), np.max(np.hstack(x_arrays))
    min_y, max_y = np.min(np.hstack(y_arrays)), np.max(np.hstack(y_arrays))

    # We want to ensure theses plots starts at 0
    if x in ["time", "n_iter"]:
        min_x = 0

    if rendering == "matplotlib":
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))
        else:
            show = False

        for i, (solver, x_array, y_array, label) in enumerate(
            zip(solvers, x_arrays, y_arrays, labels)
        ):
            color = get_plot_color(i)
            ax.plot(x_array, y_array, lw=3, label=label, color=color)

        if log_scale:
            ax.set_yscale("log")

        ax.set_xlabel(x, fontsize=16)
        ax.set_ylabel(y, fontsize=16)
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.legend()

        if show is True:
            plt.show()

        return ax.figure

    elif rendering == "bokeh":
        mins = (min_x, max_x, min_y, max_y)
        return plot_bokeh_history(
            solvers, x, y, x_arrays, y_arrays, mins, labels, log_scale, show
        )

    else:
        raise ValueError(
            "Unknown rendering type. Expected 'matplotlib' or "
            "'bokeh', received %s" % rendering
        )


def plot_bokeh_history(
    solvers, x, y, x_arrays, y_arrays, mins, legends, log_scale, show
):
    import bokeh.plotting as bk

    min_x, max_x, min_y, max_y = mins
    if log_scale:
        # Bokeh has a weird behaviour when using logscale with 0 entries...
        # We use the difference between smallest value of second small
        # to set the range of y
        all_ys = np.hstack(y_arrays)
        y_range_min = np.min(all_ys[all_ys != 0])
        if y_range_min < 0:
            raise ValueError("Cannot plot negative values on a log scale")

        fig = bk.Figure(
            plot_height=300, y_axis_type="log", y_range=[y_range_min, max_y]
        )
    else:
        fig = bk.Figure(plot_height=300, x_range=[min_x, max_x], y_range=[min_y, max_y])

    for i, (solver, x_array, y_array, legend) in enumerate(
        zip(solvers, x_arrays, y_arrays, legends)
    ):
        color = get_plot_color(i)
        fig.line(x_array, y_array, line_width=3, legend=legend, color=color)
    fig.xaxis.axis_label = x
    fig.yaxis.axis_label = y
    fig.xaxis.axis_label_text_font_size = "12pt"
    fig.yaxis.axis_label_text_font_size = "12pt"
    if show:
        bk.show(fig)
        return None
    else:
        return fig
