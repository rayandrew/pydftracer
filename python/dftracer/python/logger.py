import logging

from dftracer.python.env import *

logger: logging.Logger = setup_stream_logger()

import dftracer.python.common as common

if DFTRACER_ENABLE:
    try:
        import dftracer.dftracer as profiler  # type: ignore

        common.profiler = profiler  # type: ignore
        logger.info("DFTracer was enabled.")
    except ImportError:  # pragma: no cover
        logger.info("DFTracer was enabled but not available.")

from dftracer.python.common import *  # noqa: F403, F401, E402
