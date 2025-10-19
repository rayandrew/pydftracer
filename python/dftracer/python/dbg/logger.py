from dftracer.python.env import *

logger = setup_stream_logger()

import dftracer.python.common as common

if DFTRACER_ENABLE:
    try:
        import dftracer.dftracer_dbg as profiler

        common.profiler = profiler
        logger.info("DFTracer was enabled.")
    except ImportError:
        logger.info("DFTracer was enabled but not available.")
        pass

from dftracer.python.common import *
