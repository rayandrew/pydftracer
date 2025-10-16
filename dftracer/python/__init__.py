from importlib.metadata import version, PackageNotFoundError
from dftracer.python.logger import *
from dftracer.python.ai import *

try:
    __version__ = version("pydftracer")
except PackageNotFoundError:
    # package is not installed
    pass