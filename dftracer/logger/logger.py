import inspect
import logging
import os
import signal
import sys
from collections.abc import Iterable
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, TypeVar, Union, overload


class ProfilerProtocol(Protocol):
    """Protocol defining the interface for profiler implementations."""

    def initialize(self, log_file: str, data_dirs: str, process_id: int) -> None:
        """Initialize the profiler with configuration."""
        ...

    def get_time(self) -> int:
        """Get current time for profiling."""
        ...

    def enter_event(self) -> None:
        """Mark entry into an event."""
        ...

    def exit_event(self) -> None:
        """Mark exit from an event."""
        ...

    def log_event(
        self,
        name: str,
        cat: str,
        start_time: int,
        duration: int,
        int_args: Dict[str, Any] = {},
        string_args: Dict[str, Any] = {},
        float_args: Dict[str, float] = {},
    ) -> None:
        """Log a profiling event."""
        ...

    def log_metadata_event(self, key: str, value: str) -> None:
        """Log a metadata event."""
        ...

    def finalize(self) -> None:
        """Finalize the profiler and release resources."""
        ...


class NoOpProfiler:
    """Fallback no-operation profiler when pydftracer is not available."""

    def initialize(self, log_file: str, data_dirs: str, process_id: int) -> None:
        pass

    def get_time(self) -> int:
        return 0

    def enter_event(self) -> None:
        pass

    def exit_event(self) -> None:
        pass

    def log_event(
        self,
        name: str,
        cat: str,
        start_time: int,
        duration: int,
        int_args: Dict[str, Any] = {},
        string_args: Dict[str, Any] = {},
        float_args: Dict[str, float] = {},
    ) -> None:
        pass

    def log_metadata_event(self, key: str, value: str) -> None:
        pass

    def finalize(self) -> None:
        pass


profiler: ProfilerProtocol
try:
    import pydftracer as profiler  # type: ignore
except ImportError:
    profiler = NoOpProfiler()
    DFTRACER_ENABLE = False


DFTRACER_ENABLE_ENV = "DFTRACER_ENABLE"
DFTRACER_INIT_ENV = "DFTRACER_INIT"
DFTRACER_LOG_LEVEL_ENV = "DFTRACER_LOG_LEVEL"

DFTRACER_ENABLE = True if os.getenv(DFTRACER_ENABLE_ENV, "0") == "1" else False
DFTRACER_INIT_PRELOAD = (
    True if os.getenv(DFTRACER_INIT_ENV, "PRELOAD") == "PRELOAD" else False
)
DFTRACER_LOG_LEVEL = os.getenv(DFTRACER_LOG_LEVEL_ENV, "ERROR")

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


def capture_signal(signal_number: int, frame: Any) -> None:
    dftracer.get_instance().finalize()
    sys.exit(signal_number)


if DFTRACER_ENABLE:
    signal.signal(signal.SIGABRT, capture_signal)
    signal.signal(signal.SIGINT, capture_signal)
    signal.signal(signal.SIGTERM, capture_signal)


def setup_logger(
    name: str, log_file: str, formatter: str, level: int = logging.INFO
) -> logging.Logger:
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter(formatter))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


class dftracer:
    __instance: Optional["dftracer"] = None

    def __init__(self) -> None:
        self.logger: Optional[ProfilerProtocol] = None
        self.dbg_logging: Optional[logging.Logger] = None
        dftracer.__instance = self

    @classmethod
    def get_instance(cls) -> "dftracer":
        if dftracer.__instance is None:
            dftracer()
        # for linter (e.g. mypy) to understand that this won't be None
        assert dftracer.__instance is not None
        return dftracer.__instance

    @staticmethod
    def initialize_log(logfile: str, data_dir: str, process_id: int) -> "dftracer":
        log_file_path = None
        if logfile:
            log_file_path = Path(logfile)
        outfile = "dft.log"
        if DFTRACER_ENABLE:
            if log_file_path:
                os.makedirs(log_file_path.parent, exist_ok=True)
                outfile = os.path.join(log_file_path.parent, "dft.log")
        log_level = logging.ERROR
        if DFTRACER_LOG_LEVEL.upper() == "DEBUG":
            log_level = logging.DEBUG
        elif DFTRACER_LOG_LEVEL.upper() == "INFO":
            log_level = logging.INFO
        elif DFTRACER_LOG_LEVEL.upper() == "WARN":
            log_level = logging.WARN
        instance = dftracer.get_instance()
        instance.dbg_logging = setup_logger(
            name="dftracer_dbg",
            log_file=outfile,
            formatter="[DFTRACER_PY %(levelname)s] %(message)s [%(pathname)s:%(lineno)d]",
            level=log_level,
        )
        instance.dbg_logging.debug(
            f"logger.initialize_log {logfile} {data_dir} {process_id}"
        )
        # Use dynamic environment check for testing
        dftracer_enabled = os.getenv(DFTRACER_ENABLE_ENV, "0") == "1"
        if dftracer_enabled:
            instance.logger = profiler
            instance.dbg_logging.debug(
                f"logger.initialize {logfile} {data_dir} {process_id}"
            )
            instance.logger.initialize(
                log_file=logfile, data_dirs=data_dir, process_id=process_id
            )
        else:
            # Always use NoOpProfiler when disabled - it's defined globally
            instance.logger = NoOpProfiler()
        return instance

    def get_time(self) -> int:
        if self.logger:
            t = self.logger.get_time()
            if self.dbg_logging:
                self.dbg_logging.debug(f"logger.get_time {t}")
            return t
        return 0

    def enter_event(self) -> None:
        if self.logger:
            self.logger.enter_event()
            if self.dbg_logging:
                self.dbg_logging.debug("logger.enter_event")

    def exit_event(self) -> None:
        if self.logger:
            self.logger.exit_event()
            if self.dbg_logging:
                self.dbg_logging.debug("logger.exit_event")

    def log_event(
        self,
        name: str,
        cat: str,
        start_time: int,
        duration: int,
        string_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.logger:
            if self.dbg_logging:
                self.dbg_logging.debug(
                    f"logger.log_event {name} {cat} {start_time} {duration} {string_args}"
                )
            if string_args is None:
                string_args = {}
            self.logger.log_event(
                name=name,
                cat=cat,
                start_time=start_time,
                duration=duration,
                string_args=string_args,
            )

    def log_metadata_event(self, key: str, value: str) -> None:
        if self.logger:
            if self.dbg_logging:
                self.dbg_logging.debug(f"logger.log_metadata_event {key} {value}")
            self.logger.log_metadata_event(key=key, value=value)

    def finalize(self) -> None:
        if self.logger:
            if self.dbg_logging:
                self.dbg_logging.debug("logger.finalize")
            self.logger.finalize()


def get_default_args(func: Callable) -> Dict[str, Any]:
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class dft_fn:
    def __init__(
        self,
        cat: str,
        name: Optional[str] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ) -> None:
        self._enable = enable
        if not name:
            name = inspect.stack()[1].function
        self._name = name
        self._cat = cat
        self._arguments: Dict[str, str] = {}
        self._t1 = 0
        self._t2 = 0
        self._flush = False
        if epoch:
            self._arguments["epoch"] = str(epoch)
        if step:
            self._arguments["step"] = str(step)
        if image_idx:
            self._arguments["image_idx"] = str(image_idx)
        if image_size:
            self._arguments["image_size"] = str(image_size)
        self.reset()

    def __enter__(self) -> "dft_fn":
        if DFTRACER_ENABLE and self._enable:
            self._t1 = dftracer.get_instance().get_time()
            dftracer.get_instance().enter_event()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if DFTRACER_ENABLE and self._enable:
            dftracer.get_instance().exit_event()

    def update(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Dict[str, Any] = {},
    ) -> "dft_fn":
        if epoch is not None:
            self._arguments["epoch"] = str(epoch)
        if step is not None:
            self._arguments["step"] = str(step)
        if image_idx is not None:
            self._arguments["image_idx"] = str(image_idx)
        if image_size is not None:
            self._arguments["image_size"] = str(image_size)
        for key, value in args.items():
            self._arguments[key] = str(value)
        return self

    def flush(self) -> "dft_fn":
        if DFTRACER_ENABLE and self._enable:
            self._t2 = dftracer.get_instance().get_time()
            args = self._arguments if len(self._arguments) > 0 else None
            dftracer.get_instance().log_event(
                name=self._name,
                cat=self._cat,
                start_time=self._t1,
                duration=self._t2 - self._t1,
                string_args=args,
            )
            dftracer.get_instance().exit_event()
            self._flush = True
        return self

    def reset(self) -> "dft_fn":
        if DFTRACER_ENABLE and self._enable:
            self._t1 = dftracer.get_instance().get_time()
            dftracer.get_instance().enter_event()
            self._t2 = self._t1
        self._flush = False
        return self

    def set_flush(self, flush: bool) -> "dft_fn":
        self._flush = flush
        return self

    @overload
    def log(self, f_py: F, name: Optional[str] = None) -> F: ...

    @overload
    def log(
        self, f_py: None = None, *, name: Optional[str] = None
    ) -> Callable[[F], F]: ...

    def log(
        self, f_py: Optional[F] = None, name: Optional[str] = None
    ) -> Union[F, Callable[[F], F]]:
        # CC BY-SA 4.0 https://stackoverflow.com/a/60832711
        def _decorator(func: F) -> F:
            _name = name if name else func.__qualname__
            if DFTRACER_ENABLE and self._enable:
                arg_names = inspect.getfullargspec(func)[0]
                self._arguments = {}

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if DFTRACER_ENABLE and self._enable:
                    if len(arg_names) > 0:
                        if "self" == arg_names[0]:
                            if hasattr(args[0], "epoch"):
                                self._arguments["epoch"] = str(args[0].epoch)
                            if hasattr(args[0], "step"):
                                self._arguments["step"] = str(args[0].step)
                            if hasattr(args[0], "image_size"):
                                self._arguments["image_size"] = str(args[0].image_size)
                            if hasattr(args[0], "image_idx"):
                                self._arguments["image_idx"] = str(args[0].image_idx)
                        full_args = dict(zip(arg_names[1:], args[1:]))
                        full_args.update(kwargs)
                        full_args.update(get_default_args(func))

                        for name, value in full_args.items():
                            if name == "epoch":
                                self._arguments["epoch"] = str(value)
                            elif name == "image_idx":
                                self._arguments["image_idx"] = str(value)
                            elif name == "image_size":
                                self._arguments["image_size"] = str(value)
                            elif name == "step":
                                self._arguments["image_size"] = str(value)

                    start = dftracer.get_instance().get_time()
                    dftracer.get_instance().enter_event()
                x = func(*args, **kwargs)
                if DFTRACER_ENABLE and self._enable:
                    end = dftracer.get_instance().get_time()
                    string_args = self._arguments if len(self._arguments) > 0 else None
                    dftracer.get_instance().log_event(
                        name=_name,
                        cat=self._cat,
                        start_time=start,
                        duration=end - start,
                        string_args=string_args,
                    )
                    dftracer.get_instance().exit_event()
                return x

            return wrapper  # type: ignore

        return _decorator(f_py) if callable(f_py) else _decorator

    @overload
    def log_init(self, f_py: F, name: Optional[str] = None) -> F: ...

    @overload
    def log_init(
        self, f_py: None = None, *, name: Optional[str] = None
    ) -> Callable[[F], F]: ...

    def log_init(
        self, f_py: Optional[F] = None, name: Optional[str] = None
    ) -> Union[F, Callable[[F], F]]:
        # CC BY-SA 4.0 https://stackoverflow.com/a/60832711
        def _decorator(init: F) -> F:
            _name = name if name else init.__qualname__
            if DFTRACER_ENABLE and self._enable:
                arg_names = inspect.getfullargspec(init)[0]
                self._arguments = {}

            @wraps(init)
            def new_init(*args: Any, **kwargs: Any) -> Any:
                if DFTRACER_ENABLE and self._enable:
                    arg_values = dict(zip(arg_names[1:], args))
                    arg_values.update(kwargs)
                    arg_values.update(get_default_args(init))
                    if "epoch" in arg_values:
                        self._arguments["epoch"] = str(arg_values["epoch"])
                    elif "image_idx" in arg_values:
                        self._arguments["image_idx"] = str(arg_values["image_idx"])
                    elif "image_size" in arg_values:
                        self._arguments["image_size"] = str(arg_values["image_size"])
                    elif "step" in arg_values:
                        self._arguments["step"] = str(arg_values["step"])
                    start = dftracer.get_instance().get_time()
                    dftracer.get_instance().enter_event()

                init(*args, **kwargs)
                if DFTRACER_ENABLE and self._enable:
                    end = dftracer.get_instance().get_time()
                    string_args = self._arguments if len(self._arguments) > 0 else None
                    dftracer.get_instance().log_event(
                        name=_name,
                        cat=self._cat,
                        start_time=start,
                        duration=end - start,
                        string_args=string_args,
                    )
                    dftracer.get_instance().exit_event()

            return new_init  # type: ignore

        return _decorator(f_py) if callable(f_py) else _decorator

    @overload
    def log_static(self, f_py: F, name: Optional[str] = None) -> F: ...

    @overload
    def log_static(
        self, f_py: None = None, *, name: Optional[str] = None
    ) -> Callable[[F], F]: ...

    def log_static(
        self, f_py: Optional[F] = None, name: Optional[str] = None
    ) -> Union[F, Callable[[F], F]]:
        # CC BY-SA 4.0 https://stackoverflow.com/a/60832711
        def _decorator(func: F) -> F:
            # Extract actual function if we received a staticmethod object
            if isinstance(func, staticmethod):
                actual_func = func.__func__
            else:
                actual_func = func

            _name = name if name else func.__qualname__

            @wraps(actual_func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if DFTRACER_ENABLE and self._enable:
                    start = dftracer.get_instance().get_time()
                    dftracer.get_instance().enter_event()
                x = actual_func(*args, **kwargs)
                if DFTRACER_ENABLE and self._enable:
                    end = dftracer.get_instance().get_time()
                    string_args = self._arguments if len(self._arguments) > 0 else None
                    dftracer.get_instance().log_event(
                        name=_name,
                        cat=self._cat,
                        start_time=start,
                        duration=end - start,
                        string_args=string_args,
                    )
                    dftracer.get_instance().exit_event()
                return x

            return wrapper  # type: ignore

        # Handle both callable functions and staticmethod objects
        if f_py is not None:
            return _decorator(f_py)
        else:
            return _decorator

    def log_metadata(self, key: str, value: str) -> None:
        if DFTRACER_ENABLE and self._enable:
            dftracer.get_instance().log_metadata_event(key=key, value=value)

    def iter(
        self,
        func: Iterable[T],
        name: str = "loop",
        iter_name: str = "step",
        include_yield: bool = True,
        include_iter: bool = True,
    ) -> Iterable[T]:
        if DFTRACER_ENABLE and self._enable:
            iter_val = 1
            _name = f"{name}.iter"
            kernal_name = f"{name}.yield"
            start = dftracer.get_instance().get_time()
            self._arguments = {}

        for v in func:
            if DFTRACER_ENABLE and self._enable:
                end = dftracer.get_instance().get_time()
                t0 = dftracer.get_instance().get_time()
            yield v
            if DFTRACER_ENABLE and self._enable:
                t1 = dftracer.get_instance().get_time()
                self._arguments[iter_name] = str(iter_val)
                args = self._arguments if len(self._arguments) > 0 else None

                if include_iter:
                    dftracer.get_instance().enter_event()
                    dftracer.get_instance().log_event(
                        name=_name,
                        cat=self._cat,
                        start_time=start,
                        duration=end - start,
                        string_args=args,
                    )
                    dftracer.get_instance().exit_event()

                if include_yield:
                    dftracer.get_instance().enter_event()
                    dftracer.get_instance().log_event(
                        name=kernal_name,
                        cat=self._cat,
                        start_time=t0,
                        duration=t1 - t0,
                        string_args=args,
                    )
                    dftracer.get_instance().exit_event()

                iter_val += 1
                start = dftracer.get_instance().get_time()
