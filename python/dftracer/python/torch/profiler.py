import dftracer.python.torch.common as common
from dftracer.python.logger import dftracer

common.dftracer = dftracer  # type: ignore
from dftracer.python.torch.common import *  # type: ignore
