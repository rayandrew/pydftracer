from collections.abc import Iterable
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union, overload

# Type variables for preserving function signatures
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

DFTRACER_ENABLE_ENV = ""
DFTRACER_INIT_ENV = ""
DFTRACER_LOG_LEVEL_ENV = ""

DFTRACER_ENABLE = False
DFTRACER_INIT_PRELOAD = False
DFTRACER_LOG_LEVEL = "ERROR"


class dftracer:
    __instance = None

    def __init__(self) -> None:
        self.logger = None
        self.dbg_logging = None

    @staticmethod
    def get_instance() -> "dftracer":
        if dftracer.__instance is None:
            dftracer.__instance = dftracer()
        return dftracer.__instance

    @staticmethod
    def initialize_log(logfile: str, data_dir: str, process_id: int) -> "dftracer":
        instance = dftracer.get_instance()
        return instance

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
        string_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    def log_metadata_event(self, key: str, value: str) -> None:
        pass

    def finalize(self) -> None:
        pass


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
            name = ""
        self._name = name
        self._cat = cat
        self._arguments: Dict[str, Any] = {}
        self._flush = False
        if epoch is not None:
            self._arguments["epoch"] = str(epoch)
        if step is not None:
            self._arguments["step"] = str(step)
        if image_idx is not None:
            self._arguments["image_idx"] = str(image_idx)
        if image_size is not None:
            self._arguments["image_size"] = str(image_size)

    def __enter__(self) -> "dft_fn":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def update(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Dict[str, Any] = {},
    ) -> None:
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

    def flush(self) -> None:
        pass

    def reset(self) -> None:
        self._arguments = {}

    @overload
    def log(self, f_py: F, name: Optional[str] = None) -> F: ...

    @overload
    def log(
        self, f_py: None = None, *, name: Optional[str] = None
    ) -> Callable[[F], F]: ...

    def log(
        self, f_py: Optional[F] = None, name: Optional[str] = None
    ) -> Union[F, Callable[[F], F]]:
        def _decorator(func: F) -> F:
            self._arguments = {}

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                x = func(*args, **kwargs)
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
        def _decorator(init: F) -> F:
            @wraps(init)
            def new_init(*args: Any, **kwargs: Any) -> Any:
                init(*args, **kwargs)

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
        def _decorator(func: F) -> F:
            # Extract actual function if we received a staticmethod object
            if isinstance(func, staticmethod):
                actual_func = func.__func__
            else:
                actual_func = func

            @wraps(actual_func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return actual_func(*args, **kwargs)

            return wrapper  # type: ignore

        # Handle both callable functions and staticmethod objects
        if f_py is not None:
            return _decorator(f_py)
        else:
            return _decorator

    def log_metadata(self, key: str, value: str) -> None:
        pass

    def iter(
        self,
        func: Iterable[T],
        name: str = "loop",
        iter_name: str = "step",
        include_yield: bool = True,
        include_iter: bool = True,
    ) -> Iterable[T]:
        for v in func:
            yield v
