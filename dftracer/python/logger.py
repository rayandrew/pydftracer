from dftracer.python.env import *

import dftracer.python.common as common
if DFTRACER_ENABLE:
    import dftracer.dftracer as profiler
    common.profiler = profiler

from dftracer.python.common import *
