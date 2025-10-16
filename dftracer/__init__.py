from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("dftracer")
except PackageNotFoundError:
    # package is not installed
    pass
