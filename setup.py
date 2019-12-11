from distutils.core import setup
import io
import setuptools

CLASSIFIERS = """\
Development Status :: 1 - Planning
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 3
Topic :: Software Development
Operating System :: POSIX
Operating System :: Unix
"""

setup(
    name="linlearn",
    description="Linear learning made fast and simple",
    long_description=io.open("README.md", encoding="utf-8").read(),
    version="0.0.1",
    author="Stephane Gaiffas",
    author_email="stephane.gaiffas@gmail.com",
    url="http://pypi.python.org/pypi/linlearn",
    # packages=["linlearn"],
    packages=setuptools.find_packages(),
    install_requires=["numpy", "numba", "scipy", "tqdm", "scikit-learn"],
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    license="BSD 3-Clause License",
)
