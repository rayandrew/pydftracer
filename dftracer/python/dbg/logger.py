from dftracer.python.env import *

import dftracer.python.common
if DFTRACER_ENABLE:
    import dftracer.dftracer_dbg as profiler
    dftracer.python.common.profiler = profiler

from dftracer.python.common import *
