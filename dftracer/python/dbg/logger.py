from dftracer.python.env import *

logger = setup_stream_logger()
import dftracer.python.common
if DFTRACER_ENABLE:
    try:
        import dftracer.dftracer_dbg as profiler
        dftracer.python.common.profiler = profiler
    except ImportError:
        logger.info("DFTracer was enabled but not available.")
        pass

from dftracer.python.common import *
