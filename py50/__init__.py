from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _package_version

from .calculator import *
from .plot_settings import *
from .plotcurve import *
from .stats import *
from .utils import *

__author__ = "Tony E. Lin"
__email__ = "tonyelin@tmu.edu.tw"
__url__ = "https://github.com/tlint101/py50"

try:
    __version__ = _package_version("py50")
except _PackageNotFoundError:
    __version__ = "0.0.0+unknown"
