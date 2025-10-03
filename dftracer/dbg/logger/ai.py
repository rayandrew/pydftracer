import functools
from collections.abc import Iterator
from enum import Enum, auto
from typing import Any, Callable, Optional, TypeVar, Union, overload

from dftracer.dbg.logger.logger import DFTRACER_ENABLE, dft_fn, dftracer


# MIT License: https://github.com/irgeek/StrEnum/blob/master/strenum/__init__.py
# To support StringEnum on all python versions
class LowercaseStringEnum(str, Enum):
    def __new__(cls, value: Any, *args: Any, **kwargs: Any) -> "LowercaseStringEnum":
        # this is a specific version that will lowercase the values
        if not isinstance(value, (str, auto)):
            raise TypeError(  # pragma: no cover
                f"Values of StrEnums must be strings: {value!r} is a {type(value)}"
            )
        return super().__new__(cls, value, *args, **kwargs)

    def __str__(self) -> str:
        return str(self.value)

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[Any]
    ) -> str:
        return name.lower()


INIT_NAME = "init"
CTX_SEPARATOR = "."
ROOT_NAME = "ai_root"
ROOT_CAT = "ai_root"
ITER_NAME = "iter"
BLOCK_NAME = "block"
START_METADATA_NAME = "start_time"
STOP_METADATA_NAME = "stop_time"
ITER_COUNT_NAME = "iter_count"

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


def get_iter_block_name(name: str) -> str:
    return (
        f"{name}{CTX_SEPARATOR}{BLOCK_NAME}"
        if not name.endswith(f"{CTX_SEPARATOR}{BLOCK_NAME}")
        else name
    )


def get_iter_handle_name(name: str) -> str:
    return (
        f"{name}{CTX_SEPARATOR}{ITER_NAME}"
        if not name.endswith(f"{CTX_SEPARATOR}{ITER_NAME}")
        else name
    )


class _DFTracerAI:
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
        if not name:
            name = cat

        self.profiler = dft_fn(
            cat=cat,
            name=name,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )

    @overload
    def __call__(
        self,
        fn: F,
        *,
        enable: Optional[bool] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[dict[str, Any]] = None,
    ) -> F: ...

    @overload
    def __call__(
        self,
        *,
        enable: Optional[bool] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[dict[str, Any]] = None,
    ) -> "DFTracerAI": ...

    def __call__(
        self,
        fn: Optional[F] = None,
        *,
        enable: Optional[bool] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[dict[str, Any]] = None,
    ) -> Union[F, "DFTracerAI"]:
        if epoch is not None:
            self.profiler._arguments["epoch"] = str(epoch)
        if step is not None:
            self.profiler._arguments["step"] = str(step)
        if image_idx is not None:
            self.profiler._arguments["image_idx"] = str(image_idx)
        if image_size is not None:
            self.profiler._arguments["image_size"] = str(image_size)
        args = args or {}
        for key, value in args.items():
            self.profiler._arguments[key] = str(value)

        is_enabled = self.profiler._enable if enable is None else enable

        if fn:

            def _decorator(f: F) -> F:
                @functools.wraps(f)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    if is_enabled:
                        with self:
                            return f(*args, **kwargs)
                    return f(*args, **kwargs)

                return wrapper  # type: ignore

            return _decorator(fn)
        else:
            return DFTracerAI(
                cat=self.profiler._cat,
                name=self.profiler._name,
                epoch=epoch,
                step=step,
                image_idx=image_idx,
                image_size=image_size,
                enable=is_enabled,
            )

    def __enter__(self) -> "_DFTracerAI":
        self.profiler.__enter__()
        # Reset flush state to ensure proper event logging on each context manager entry.
        # The underlying DFTracer logger was designed for one-time use objects, but this
        # class acts as a singleton. Without this reset, events won't flush after the
        # first __exit__ call due to DFTracer's internal _flush state management.
        self.profiler.set_flush(False)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.profiler.__exit__(exc_type, exc_val, exc_tb)

    def start(self, metadata: bool = False) -> None:
        if metadata:
            time = dftracer.get_instance().get_time()
            self.profiler.log_metadata(
                key=f"{self.profiler._name}{CTX_SEPARATOR}{START_METADATA_NAME}",
                value=str(time),
            )
        else:
            self.__enter__()

    def stop(self, metadata: bool = False) -> None:
        if metadata:
            time = dftracer.get_instance().get_time()
            self.profiler.log_metadata(
                key=f"{self.profiler._name}{CTX_SEPARATOR}{STOP_METADATA_NAME}",
                value=str(time),
            )
        else:
            self.__exit__(None, None, None)

    def enable(self) -> None:
        self.profiler._enable = True

    def disable(self) -> None:
        self.profiler._enable = False

    @property
    def cat(self) -> str:
        return self.profiler._cat

    @property
    def name(self) -> str:
        return self.profiler._name

    def update(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[dict[str, Any]] = None,
    ) -> "_DFTracerAI":
        if args is None:
            args = {}
        if DFTRACER_ENABLE and self.profiler._enable:
            if epoch is not None:
                self.profiler._arguments["epoch"] = str(epoch)
            if step is not None:
                self.profiler._arguments["step"] = str(step)
            if image_idx is not None:
                self.profiler._arguments["image_idx"] = str(image_idx)
            if image_size is not None:
                self.profiler._arguments["image_size"] = str(image_size)
            for key, value in args.items():
                self.profiler._arguments[key] = str(value)
        return self

    def iter(
        self,
        iterator: Iterator[T],
        *,
        include_block: bool = True,
        include_iter: bool = True,
        iter_name: Optional[str] = None,
        block_name: Optional[str] = None,
    ) -> Iterator[T]:
        iter_name = iter_name or get_iter_handle_name(self.profiler._name)
        block_name = block_name or get_iter_block_name(self.profiler._name)
        iter_val = 1

        start: int = 0
        if DFTRACER_ENABLE and self.profiler._enable:
            self.profiler._arguments = {}
            start = dftracer.get_instance().get_time()

        for v in iterator:
            if DFTRACER_ENABLE and self.profiler._enable:
                end = dftracer.get_instance().get_time()
                t0 = dftracer.get_instance().get_time()

            yield v

            if DFTRACER_ENABLE and self.profiler._enable:
                t1 = dftracer.get_instance().get_time()
                self.profiler._arguments[ITER_COUNT_NAME] = str(iter_val)
                args = (
                    self.profiler._arguments
                    if len(self.profiler._arguments) > 0
                    else None
                )

                if include_iter:
                    dftracer.get_instance().enter_event()
                    dftracer.get_instance().log_event(
                        name=iter_name,
                        cat=self.profiler._cat,
                        start_time=start,
                        duration=end - start,
                        string_args=args,
                    )
                    dftracer.get_instance().exit_event()

                if include_block:
                    dftracer.get_instance().enter_event()
                    dftracer.get_instance().log_event(
                        name=block_name,
                        cat=self.profiler._cat,
                        start_time=t0,
                        duration=t1 - t0,
                        string_args=args,
                    )
                    dftracer.get_instance().exit_event()

                iter_val += 1
                start = dftracer.get_instance().get_time()


class DFTracerAI(_DFTracerAI):
    def __init__(
        self,
        *,
        cat: str,
        name: Optional[str] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(
            cat=cat,
            name=name,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self._children: dict[str, DFTracerAI] = {}

    def create_children(self, names: dict[str, str]) -> None:
        for attr, name in names.items():
            tracer = DFTracerAI(
                cat=self.profiler._cat,
                name=name,
                epoch=int(self.profiler._arguments["epoch"])
                if "epoch" in self.profiler._arguments
                else None,
                step=int(self.profiler._arguments["step"])
                if "step" in self.profiler._arguments
                else None,
                image_idx=int(self.profiler._arguments["image_idx"])
                if "image_idx" in self.profiler._arguments
                else None,
                image_size=self.profiler._arguments.get("image_size"),
                enable=self.profiler._enable,
            )
            setattr(self, attr, tracer)
            self._children[attr] = tracer

    def update(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[dict[str, Any]] = None,
    ) -> "DFTracerAI":
        super().update(
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            args=args,
        )
        for tracer in self._children.values():
            tracer.update(
                epoch=epoch,
                step=step,
                image_idx=image_idx,
                image_size=image_size,
                args=args,
            )
        return self

    def enable(self) -> None:
        super().enable()
        for tracer in self._children.values():
            tracer.enable()

    def disable(self) -> None:
        super().disable()
        for tracer in self._children.values():
            tracer.disable()

    def derive(self, name: str) -> "DFTracerAI":
        _name = f"{self.profiler._name}{CTX_SEPARATOR}{name}"
        if _name in self._children:
            return self._children[_name]

        child = DFTracerAI(
            cat=self.profiler._cat,
            name=_name,
            epoch=int(self.profiler._arguments["epoch"])
            if "epoch" in self.profiler._arguments
            else None,
            step=int(self.profiler._arguments["step"])
            if "step" in self.profiler._arguments
            else None,
            image_idx=int(self.profiler._arguments["image_idx"])
            if "image_idx" in self.profiler._arguments
            else None,
            image_size=self.profiler._arguments.get("image_size"),
            enable=self.profiler._enable,
        )
        self._children[_name] = child
        return child

    @overload
    def init(
        self,
        fn: F,
        *,
        enable: Optional[bool] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[dict[str, Any]] = None,
    ) -> F: ...

    @overload
    def init(
        self,
        fn: None = None,
        *,
        enable: Optional[bool] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[dict[str, Any]] = None,
    ) -> "DFTracerAI": ...

    def init(
        self,
        fn: Optional[F] = None,
        *,
        enable: Optional[bool] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[dict[str, Any]] = None,
    ) -> Union[F, "DFTracerAI"]:
        _name = f"{self.profiler._name}{CTX_SEPARATOR}{INIT_NAME}"
        if _name in self._children:
            if fn is not None:
                return self._children[_name](
                    fn=fn,
                    enable=enable,
                    epoch=epoch,
                    step=step,
                    image_idx=image_idx,
                    image_size=image_size,
                    args=args,
                )
            else:
                return self._children[_name](
                    enable=enable,
                    epoch=epoch,
                    step=step,
                    image_idx=image_idx,
                    image_size=image_size,
                    args=args,
                )

        child = DFTracerAI(
            cat=self.profiler._cat,
            name=_name,
            epoch=int(self.profiler._arguments["epoch"])
            if "epoch" in self.profiler._arguments
            else None,
            step=int(self.profiler._arguments["step"])
            if "step" in self.profiler._arguments
            else None,
            image_idx=int(self.profiler._arguments["image_idx"])
            if "image_idx" in self.profiler._arguments
            else None,
            image_size=self.profiler._arguments.get("image_size"),
            enable=self.profiler._enable,
        )
        self._children[_name] = child
        if fn is not None:
            return child(
                fn=fn,
                enable=enable,
                epoch=epoch,
                step=step,
                image_idx=image_idx,
                image_size=image_size,
                args=args,
            )
        else:
            return child(
                enable=enable,
                epoch=epoch,
                step=step,
                image_idx=image_idx,
                image_size=image_size,
                args=args,
            )


# Enumerations


class ProfileCategory(LowercaseStringEnum):
    COMPUTE = auto()
    DATA = auto()
    DATALOADER = auto()
    COMM = auto()
    DEVICE = auto()
    CHECKPOINT = auto()
    PIPELINE = auto()


class ComputeEvent(LowercaseStringEnum):
    FORWARD = auto()
    BACKWARD = auto()
    STEP = auto()


class DataEvent(LowercaseStringEnum):
    PREPROCESS = auto()
    ITEM = auto()


class DataLoaderEvent(LowercaseStringEnum):
    FETCH = auto()


# Terminologies are taken from https://docs.pytorch.org/docs/stable/distributed.html
class CommunicationEvent(LowercaseStringEnum):
    SEND = auto()
    RECEIVE = auto()
    BARRIER = auto()
    BCAST = auto()
    REDUCE = auto()
    ALL_REDUCE = auto()
    GATHER = auto()
    ALL_GATHER = auto()
    SCATTER = auto()
    REDUCE_SCATTER = auto()
    ALL_TO_ALL = auto()


class DeviceEvent(LowercaseStringEnum):
    TRANSFER = auto()


class PipelineEvent(LowercaseStringEnum):
    EPOCH = auto()
    TRAIN = auto()
    EVALUATE = auto()
    TEST = auto()


class CheckpointEvent(LowercaseStringEnum):
    CAPTURE = auto()
    RESTART = auto()


class _Compute(DFTracerAI):
    forward: DFTracerAI
    backward: DFTracerAI
    step: DFTracerAI

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ) -> None:
        super().__init__(
            cat=ProfileCategory.COMPUTE,
            name=ProfileCategory.COMPUTE,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )

        self.create_children(
            {
                "forward": ComputeEvent.FORWARD,
                "backward": ComputeEvent.BACKWARD,
                "step": ComputeEvent.STEP,
            }
        )


class _Data(DFTracerAI):
    preprocess: DFTracerAI
    item: DFTracerAI

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(
            cat=ProfileCategory.DATA,
            name=ProfileCategory.DATA,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self.create_children(
            {
                "preprocess": DataEvent.PREPROCESS,
                "item": DataEvent.ITEM,
            }
        )


class _DataLoader(DFTracerAI):
    fetch: DFTracerAI

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(
            cat=ProfileCategory.DATALOADER,
            name=ProfileCategory.DATALOADER,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self.create_children(
            {
                "fetch": DataLoaderEvent.FETCH,
            }
        )


class _Communication(DFTracerAI):
    send: DFTracerAI
    receive: DFTracerAI
    barrier: DFTracerAI
    bcast: DFTracerAI
    reduce: DFTracerAI
    all_reduce: DFTracerAI
    gather: DFTracerAI
    all_gather: DFTracerAI
    scatter: DFTracerAI
    reduce_scatter: DFTracerAI
    all_to_all: DFTracerAI

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(
            cat=ProfileCategory.COMM,
            name=ProfileCategory.COMM,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self.create_children(
            {
                "send": CommunicationEvent.SEND,
                "receive": CommunicationEvent.RECEIVE,
                "barrier": CommunicationEvent.BARRIER,
                "bcast": CommunicationEvent.BCAST,
                "reduce": CommunicationEvent.REDUCE,
                "all_reduce": CommunicationEvent.ALL_REDUCE,
                "gather": CommunicationEvent.GATHER,
                "all_gather": CommunicationEvent.ALL_GATHER,
                "scatter": CommunicationEvent.SCATTER,
                "reduce_scatter": CommunicationEvent.REDUCE_SCATTER,
                "all_to_all": CommunicationEvent.ALL_TO_ALL,
            }
        )


class _Device(DFTracerAI):
    transfer: DFTracerAI

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(
            cat=ProfileCategory.DEVICE,
            name=ProfileCategory.DEVICE,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self.create_children(
            {
                "transfer": DeviceEvent.TRANSFER,
            }
        )


class _Checkpoint(DFTracerAI):
    capture: DFTracerAI
    restart: DFTracerAI

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(
            cat=ProfileCategory.CHECKPOINT,
            name=ProfileCategory.CHECKPOINT,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self.create_children(
            {
                "capture": CheckpointEvent.CAPTURE,
                "restart": CheckpointEvent.RESTART,
            }
        )


class _Pipeline(DFTracerAI):
    epoch: DFTracerAI
    train: DFTracerAI
    evaluate: DFTracerAI
    test: DFTracerAI

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(
            cat=ProfileCategory.PIPELINE,
            name=ProfileCategory.PIPELINE,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self.create_children(
            {
                "epoch": PipelineEvent.EPOCH,
                "train": PipelineEvent.TRAIN,
                "evaluate": PipelineEvent.EVALUATE,
                "test": PipelineEvent.TEST,
            }
        )


# fmt: off
class _AI(DFTracerAI):
    compute: _Compute
    data: _Data
    dataloader: _DataLoader
    comm: _Communication
    device: _Device
    checkpoint: _Checkpoint
    pipeline: _Pipeline

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(cat=ROOT_CAT, name=ROOT_NAME, epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)
        self.compute = _Compute(epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)
        self.data = _Data(epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)
        self.dataloader = _DataLoader(epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)
        self.comm = _Communication(epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)
        self.device = _Device(epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)
        self.checkpoint = _Checkpoint(epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)
        self.pipeline = _Pipeline(epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)

        self._children = {
            "compute": self.compute,
            "data": self.data,
            "dataloader": self.dataloader,
            "comm": self.comm,
            "device": self.device,
            "checkpoint": self.checkpoint,
            "pipeline": self.pipeline,
        }
# fmt: on


ai = _AI()
comm = ai.comm
compute = ai.compute
data = ai.data
dataloader = ai.dataloader
device = ai.device
checkpoint = ai.checkpoint
pipeline = ai.pipeline

__all__ = [
    "INIT_NAME",
    "ITER_COUNT_NAME",
    "CommunicationEvent",
    "ComputeEvent",
    "DataEvent",
    "DataLoaderEvent",
    "DeviceEvent",
    "PipelineEvent",
    "ProfileCategory",
    "ai",
    "comm",
    "compute",
    "data",
    "dataloader",
    "device",
    "get_iter_block_name",
    "get_iter_handle_name",
    "pipeline",
]
