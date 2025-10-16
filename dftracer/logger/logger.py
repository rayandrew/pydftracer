from dftracer.logger.env import *

import dftracer.logger.common as common
if DFTRACER_ENABLE:
    import dftracer as profiler
    common.profiler = profiler

from dftracer.logger.common import *
