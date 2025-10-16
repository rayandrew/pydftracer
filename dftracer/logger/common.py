
from dftracer.logger.env import *

profiler = None

from functools import wraps
from typing import Dict
import logging


from pathlib import Path
import inspect
import sys, signal
from enum import Enum

class TagType(Enum):
    KEY = 0
    VALUE = 1
    IGNORE = 2

    def __int__(self):
        return self.value

class TagDType(Enum):
    INT = 0
    FLOAT = 1
    STRING = 2

    def __int__(self):
        return self.value

class TagValue:
    def __init__(self, value, dtype: TagDType = TagDType.STRING, tag_type: TagType = TagType.KEY):
        self._dtype = dtype
        self._value = value
        self._tag_type = tag_type
        if self._dtype == TagDType.INT:
            self._value = int(value)
        elif self._dtype == TagDType.FLOAT:
            self._value = float(value)
        else:
            self._value = str(value)

    def value(self):
        return (int(self._tag_type), self._value)

def capture_signal(signal_number, frame):
    dftracer.get_instance().finalize()
    sys.exit(signal_number)

if DFTRACER_ENABLE:
    signal.signal(signal.SIGABRT, capture_signal)
    signal.signal(signal.SIGINT, capture_signal)
    signal.signal(signal.SIGTERM, capture_signal)


def setup_logger(name, log_file, formatter, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter(formatter))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

class dftracer:
    __instance = None

    def __init__(self):
        if DFTRACER_ENABLE:
            self.logger = None
            self.dbg_logging = None
        dftracer.__instance = self

    @classmethod
    def get_instance(cls):
        """ Static access method. """
        if dftracer.__instance is None:
            dftracer()
        return dftracer.__instance

    @staticmethod
    def initialize_log(logfile, data_dir, process_id):
        global profiler
        log_file_path = None
        if logfile:
            log_file_path = Path(logfile)
        outfile = "dft.log"
        if DFTRACER_ENABLE:
            if log_file_path:
                os.makedirs(log_file_path.parent, exist_ok=True)
                outfile = os.path.join(log_file_path.parent, "dft.log")
        log_level = logging.ERROR
        if DFTRACER_LOG_LEVEL == "DEBUG":
            log_level = logging.DEBUG
        elif DFTRACER_LOG_LEVEL == "INFO":
            log_level = logging.INFO
        elif DFTRACER_LOG_LEVEL == "WARN":
            log_level = logging.WARN
        instance = dftracer.get_instance()
        instance.dbg_logging = setup_logger(name="dftracer_dbg", log_file=outfile, formatter='[DFTRACER_PY %(levelname)s] %(message)s [%(pathname)s:%(lineno)d]', level=log_level)
        instance.dbg_logging.debug(f"logger.initialize_log {logfile} {data_dir} {process_id}")
        if DFTRACER_ENABLE:
            instance.logger = profiler
            instance.dbg_logging.debug(f"logger.initialize {logfile} {data_dir} {process_id}")
            instance.logger.initialize(log_file=logfile, data_dirs=data_dir, process_id=process_id)
        return instance

    def get_time(self):
        if DFTRACER_ENABLE and self.logger:
            t = self.logger.get_time()
            self.dbg_logging.debug(f"logger.get_time {t}")
            return t
        return 0

    def enter_event(self):
        if DFTRACER_ENABLE and self.logger:
            self.logger.enter_event()
            self.dbg_logging.debug(f"logger.enter_event")

    def exit_event(self):
        if DFTRACER_ENABLE and self.logger:
            self.logger.exit_event()
            self.dbg_logging.debug(f"logger.exit_event")

    def log_event(self, name, cat, start_time, duration, int_args={}, float_args={}, string_args={}):
        if DFTRACER_ENABLE and self.logger:
            self.dbg_logging.debug(f"logger.log_event {name} {cat} {start_time} {duration} {string_args}")
            self.logger.log_event(name=name, cat=cat, start_time=start_time, duration=duration, int_args=int_args, float_args=float_args, string_args=string_args)

    def log_metadata_event(self, key, value):
        if DFTRACER_ENABLE and self.logger:
            self.dbg_logging.debug(f"logger.log_metadata_event {key} {value}")
            self.logger.log_metadata_event(key=key, value=value)

    def finalize(self):
        if DFTRACER_ENABLE and self.logger:
            self.dbg_logging.debug(f"logger.finalize")
            self.logger.finalize()

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

class dft_fn(object):

    def __init__(self, cat, name=None, epoch=None, step=None, image_idx=None, image_size=None, enable=True):
        self._enable = enable
        self._name = name
        self._cat = cat
        self._arguments_int: Dict[str, tuple[int, int]] = {}
        self._arguments_float: Dict[str, tuple[int, float]] = {}
        self._arguments_string: Dict[str, tuple[int, str]] = {}
        if DFTRACER_ENABLE and self._enable:
            if not name:
                name = inspect.stack()[1].function
            if epoch is not None: self._arguments_int["epoch"] = TagValue(epoch, TagDType.INT, TagType.KEY).value()
            if step is not None: self._arguments_int["step"] = TagValue(step, TagDType.INT, TagType.KEY).value()
            if image_idx is not None: self._arguments_int["image_idx"] = TagValue(image_idx, TagDType.INT, TagType.KEY).value()
            if image_size is not None: self._arguments_float["image_size"] = TagValue(image_size, TagDType.FLOAT, TagType.KEY).value()
            self.reset()

    def __enter__(self):
        if DFTRACER_ENABLE and self._enable:
            self._t1 = dftracer.get_instance().get_time()
            dftracer.get_instance().enter_event()
        return self

    def update(self, epoch=None, step=None, image_idx=None, image_size=None, args={}):
        if DFTRACER_ENABLE and self._enable:
            if epoch is not None: self._arguments_int["epoch"] = TagValue(epoch, TagDType.INT, TagType.KEY).value()
            if step is not None: self._arguments_int["step"] = TagValue(step, TagDType.INT, TagType.KEY).value()
            if image_idx is not None: self._arguments_int["image_idx"] = TagValue(image_idx, TagDType.INT, TagType.KEY).value()
            if image_size is not None: self._arguments_float["image_size"] = TagValue(image_size, TagDType.FLOAT, TagType.KEY).value()
            for key, value in args.items():
                if isinstance(value, TagValue):
                    new_value = value._value
                else:
                    new_value = value
                if isinstance(new_value, int):
                    self._arguments_int[key] = TagValue(new_value, TagDType.INT, TagType.KEY).value()
                elif isinstance(new_value, float):
                    self._arguments_float[key] = TagValue(new_value, TagDType.FLOAT, TagType.KEY).value()
                else:
                    self._arguments_string[key] = TagValue(str(new_value), TagDType.STRING, TagType.KEY).value()
        return self

    def flush(self):
        if DFTRACER_ENABLE and self._enable:
            self._t2 = dftracer.get_instance().get_time()
            dftracer.get_instance().log_event(name=self._name, cat=self._cat, start_time=self._t1,
                                                 duration=self._t2 - self._t1,
                                                 int_args=self._arguments_int,
                                                 float_args=self._arguments_float,
                                                 string_args=self._arguments_string)
            dftracer.get_instance().exit_event()
            self._flush = True
        return self

    def reset(self):
        if DFTRACER_ENABLE and self._enable:
            self._t1 = dftracer.get_instance().get_time()
            dftracer.get_instance().enter_event()
            self._t2 = self._t1
            self._flush = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if DFTRACER_ENABLE and self._enable:
            if not self._flush:
                self.flush()

    def log(self, func):
        if DFTRACER_ENABLE and self._enable:
            arg_names = inspect.getfullargspec(func)[0]
            self._arguments_int: Dict[str, tuple[int, int]] = {}
            self._arguments_float: Dict[str, tuple[int, float]] = {}
            self._arguments_string: Dict[str, tuple[int, str]] = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            if DFTRACER_ENABLE and self._enable:
                if len(arg_names) > 0:
                    if "self" == arg_names[0]:
                        if hasattr(args[0], "epoch"):
                            self._arguments_int["epoch"] = TagValue(args[0].epoch, TagDType.INT, TagType.KEY).value()
                        if hasattr(args[0], "step"):
                            self._arguments_int["step"] = TagValue(args[0].step, TagDType.INT, TagType.KEY).value()
                        if hasattr(args[0], "image_size"):
                            self._arguments_float["image_size"] = TagValue(args[0].image_size, TagDType.FLOAT, TagType.KEY).value()
                        if hasattr(args[0], "image_idx"):
                            self._arguments_int["image_idx"] = TagValue(args[0].image_idx, TagDType.INT, TagType.KEY).value()
                    full_args = dict(zip(arg_names[1:], args[1:]))
                    full_args.update(kwargs)
                    full_args.update(get_default_args(func))

                    for name, value in full_args.items():
                        if name == "epoch":
                            self._arguments_int["epoch"] = TagValue(value, TagDType.INT, TagType.KEY).value()
                        elif name == "image_idx":
                            self._arguments_int["image_idx"] = TagValue(value, TagDType.INT, TagType.KEY).value()
                        elif name == "image_size":
                            self._arguments_float["image_size"] = TagValue(value, TagDType.FLOAT, TagType.KEY).value()
                        elif name == "step":
                            self._arguments_int["step"] = TagValue(value, TagDType.INT, TagType.KEY).value()

                start = dftracer.get_instance().get_time()
                dftracer.get_instance().enter_event()
            x = func(*args, **kwargs)
            if DFTRACER_ENABLE and self._enable:
                end = dftracer.get_instance().get_time()
                dftracer.get_instance().log_event(name=func.__qualname__, cat=self._cat, start_time=start,
                                                        duration=end - start,
                                                        int_args=self._arguments_int,
                                                        float_args=self._arguments_float,
                                                        string_args=self._arguments_string)
                dftracer.get_instance().exit_event()
            return x

        return wrapper

    def log_metadata(self, key, value):
        if DFTRACER_ENABLE and self._enable:
            dftracer.get_instance().log_metadata_event(key=key, value=value)

    def iter(self, func, name="loop", iter_name="step"):
        if DFTRACER_ENABLE and self._enable:
            iter_val = 1
            _name = f"{name}.iter"
            kernal_name = f"{name}.yield"
            start = dftracer.get_instance().get_time()
            self._arguments_int: Dict[str, tuple[int, int]] = {}
            self._arguments_float: Dict[str, tuple[int, float]] = {}
            self._arguments_string: Dict[str, tuple[int, str]] = {}

        for v in func:
            if DFTRACER_ENABLE and self._enable:
                end = dftracer.get_instance().get_time()
                t0 = dftracer.get_instance().get_time()
            yield v
            if DFTRACER_ENABLE and self._enable:
                t1 = dftracer.get_instance().get_time()
                self._arguments_int[iter_name] = TagValue(iter_val, TagDType.INT, TagType.KEY).value()
                dftracer.get_instance().enter_event()
                dftracer.get_instance().log_event(name=_name, cat=self._cat, start_time=start,
                                                        duration=end - start,
                                                        int_args=self._arguments_int,
                                                        float_args=self._arguments_float,
                                                        string_args=self._arguments_string)
                dftracer.get_instance().exit_event()
                dftracer.get_instance().enter_event()
                dftracer.get_instance().log_event(name=kernal_name, cat=self._cat, start_time=t0,
                                                        duration=t1 - t0,
                                                        int_args=self._arguments_int,
                                                        float_args=self._arguments_float,
                                                        string_args=self._arguments_string)
                dftracer.get_instance().exit_event()

                iter_val += 1
                start = dftracer.get_instance().get_time()

    def log_init(self, init):
        if DFTRACER_ENABLE and self._enable:
            arg_names = inspect.getfullargspec(init)[0]
            self._arguments_int: Dict[str, tuple[int, int]] = {}
            self._arguments_float: Dict[str, tuple[int, float]] = {}
            self._arguments_string: Dict[str, tuple[int, str]] = {}

        @wraps(init)
        def new_init(*args, **kwargs):
            if DFTRACER_ENABLE and self._enable:
                arg_values = dict(zip(arg_names, args))
                arg_values.update(kwargs)
                arg_values.update(get_default_args(init))
                if "epoch" in arg_values:
                    self._arguments_int["epoch"] = TagValue(arg_values["epoch"], TagDType.INT, TagType.KEY).value()
                elif "image_idx" in arg_values:
                    self._arguments_int["image_idx"] = TagValue(arg_values["image_idx"], TagDType.INT, TagType.KEY).value()
                elif "image_size" in arg_values:
                    self._arguments_float["image_size"] = TagValue(arg_values["image_size"], TagDType.FLOAT, TagType.KEY).value()
                elif "step" in arg_values:
                    self._arguments_int["step"] = TagValue(arg_values["step"], TagDType.INT, TagType.KEY).value()
                start = dftracer.get_instance().get_time()
                dftracer.get_instance().enter_event()
            init(*args, **kwargs)
            if DFTRACER_ENABLE and self._enable:
                end = dftracer.get_instance().get_time()
                dftracer.get_instance().log_event(name=init.__qualname__, cat=self._cat, start_time=start,
                                                        duration=end - start,
                                                        int_args=self._arguments_int,
                                                        float_args=self._arguments_float,
                                                        string_args=self._arguments_string)
                dftracer.get_instance().exit_event()
        return new_init

    def log_static(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if DFTRACER_ENABLE and self._enable:
                start = dftracer.get_instance().get_time()
                dftracer.get_instance().enter_event()
            x = func(*args, **kwargs)
            if DFTRACER_ENABLE and self._enable:
                end = dftracer.get_instance().get_time()
                dftracer.get_instance().log_event(name=func.__qualname__, cat=self._cat, start_time=start,
                                                        duration=end - start,
                                                        int_args=self._arguments_int,
                                                        float_args=self._arguments_float,
                                                        string_args=self._arguments_string)
                dftracer.get_instance().exit_event()
            return x

        return wrapper