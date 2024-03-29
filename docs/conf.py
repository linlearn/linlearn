# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

# import sphinx_readable_theme
# html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
# html_theme = 'readable'

html_theme = 'sphinx_rtd_theme'

# html_theme = 'python_docs_theme'

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("sphinx_ext"))

# from linlearn import *
# from github_link import make_linkcode_resolve

# -- Project information -----------------------------------------------------

project = "linlearn"
copyright = "2020, Stéphane Gaïffas"
author = "Stéphane Gaïffas"

# The full version, including alpha/beta/rc tags
release = "0.0.1"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    # "sphinx.ext.linkcode",
    #    "sphinx_gallery.gen_gallery",
]

# autosummary_generate = True


autoclass_content = "class"
autodoc_inherit_docstrings = True

autodoc_default_flags = "inherited-members"

autodoc_default_options = {
    # "members": None,
    "member-order": "bysource",
    # "inherited-members": None,
    "autoclass_content": "class",
}


# sphinx_gallery_conf = {
#     "examples_dirs": "../examples",
#     "doc_module": "linlearn",
#     "gallery_dirs": "auto_examples",
#     "ignore_pattern": "../run_*|../playground_*",
#     "backreferences_dir": os.path.join("modules", "generated"),
#     "show_memory": False,
#     "reference_url": {"onelearn": None},
# }


# linkcode_resolve = make_linkcode_resolve(
#     "linlearn",
#     u"https://github.com/linlearn/"
#     "linlearn/blob/{revision}/"
#     "{package}/{path}#L{lineno}",
# )

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = ".rst"

# Generate the plots for the gallery
plot_gallery = "True"

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "press"
# html_theme = "readable"

# html_sidebars = {
#     "**": ["about.html", "navigation.html", "searchbox.html"],
#     "auto_examples": ["index.html"],
# }
html_theme_options = {
    # "description": "Linear methods in Python",
    # "github_user": "linlearn",
    # "github_repo": "linlearn",
    # "github_button": True,
    # "fixed_sidebar": True,
    # "travis_button": False,
    # "logo_text_align": "center",
    # "github_banner": True,
}

# html_logo = "images/logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# from datetime import datetime

# now = datetime.now()
# html_show_copyright = copyright = (
#     str(now.year)
#     + ', <a href="https://github.com/linlearn/linlearn/graphs/contributors">linlearn developers</a>. Updated on '
#     + now.strftime("%B %d<, %Y")
# )


# intersphinx_mapping = {
#     "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
#     "numpy": ("https://docs.scipy.org/doc/numpy/", None),
#     "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
#     "matplotlib": ("https://matplotlib.org/", None),
#     "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
#     "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
#     "sklearn": ("https://scikit-learn.org/stable/", None),
# }
