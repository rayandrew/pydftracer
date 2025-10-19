import logging
import os

DFTRACER_ENABLE_ENV = "DFTRACER_ENABLE"
DFTRACER_INIT_ENV = "DFTRACER_INIT"
DFTRACER_LOG_LEVEL_ENV = "DFTRACER_LOG_LEVEL"

DFTRACER_ENABLE = True if os.getenv(DFTRACER_ENABLE_ENV, "0") == "1" else False
DFTRACER_INIT_PRELOAD = (
    True if os.getenv(DFTRACER_INIT_ENV, "PRELOAD") == "PRELOAD" else False
)
DFTRACER_LOG_LEVEL = os.getenv(DFTRACER_LOG_LEVEL_ENV, "ERROR")


def setup_stream_logger() -> logging.Logger:
    log_level = logging.ERROR
    if DFTRACER_LOG_LEVEL == "DEBUG":
        log_level = logging.DEBUG
    elif DFTRACER_LOG_LEVEL == "INFO":
        log_level = logging.INFO
    elif DFTRACER_LOG_LEVEL == "WARN":
        log_level = logging.WARN
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger = logging.getLogger("DFTRACER")
    logger.setLevel(log_level)
    logger.addHandler(handler)
    return logger
