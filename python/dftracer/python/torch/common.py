from typing import Any

from dftracer.python.common import TagDType, TagType, TagValue, get_time_scale

dftracer = None  # type: ignore


# 1. Custom Profiler Plugin (handler)
def trace_handler(profiler_result: Any) -> None:
    global dftracer
    events = profiler_result.events()

    # event.time_range.start/end are PyTorch-profiler-relative MICROSECONDS
    # (kineto_event.start_ns() - trace_start_ns, see torch/autograd/profiler.py),
    # i.e. near-zero offsets from when this profiler session started — NOT the
    # same clock/unit as dftracer.get_instance().get_time(), which is
    # wall-clock time in whatever DFTRACER_TIME_METRIC selects (default
    # microseconds, but NS/MS/SEC are all valid) and is what every other
    # event category is logged in. Logging the raw relative value here
    # plants "PP" events near time 0, on the wrong unit besides, while the
    # rest of the trace sits at the real epoch — they show up as a separate
    # island far outside the run's actual timeline.
    #
    # trace_start_ns() is kineto's own absolute epoch-NANOSECOND timestamp
    # for this profiler session's relative zero point. get_time_scale()
    # reports dftracer's current units-per-second (from DFTRACER_TIME_METRIC)
    # so both the anchor and each relative offset can be converted into
    # dftracer's own clock/unit generically, whatever it's configured to be.
    scale = get_time_scale()  # dftracer time units per second
    trace_start_ns = profiler_result.profiler.kineto_results.trace_start_ns()
    trace_start_dft = trace_start_ns * scale / 1e9
    us_to_dft = scale / 1e6  # dftracer time units per PyTorch-profiler microsecond

    # Print attributes for each event
    dftracer.get_instance().enter_event()  # type: ignore
    for _i, event in enumerate(events):
        # Extract kernel name from event.key
        key = event.key
        # Check available attributes of time_range
        start_time = int(trace_start_dft + event.time_range.start * us_to_dft)
        duration = int(event.time_range.elapsed_us() * us_to_dft)
        int_args = {}
        int_args["device"] = TagValue(
            event.device_type, TagDType.INT, TagType.KEY
        ).value()
        int_args["cpu_memory"] = TagValue(
            event.cpu_memory_usage, TagDType.INT, TagType.KEY
        ).value()
        int_args["is_remote"] = TagValue(
            event.is_remote, TagDType.INT, TagType.KEY
        ).value()
        int_args["device_memory_usage"] = TagValue(
            event.device_memory_usage, TagDType.INT, TagType.KEY
        ).value()
        int_args["input_size"] = TagValue(
            sum(len(s) for s in event.input_shapes) if event.input_shapes else 0,
            TagDType.INT,
            TagType.KEY,
        ).value()
        float_args = {}
        float_args["total_cpu_percent"] = TagValue(
            event.total_cpu_percent, TagDType.FLOAT, TagType.KEY
        ).value()
        float_args["total_device_percent"] = TagValue(
            event.total_device_percent, TagDType.FLOAT, TagType.KEY
        ).value()

        dftracer.get_instance().log_event(  # type: ignore
            name=key,
            cat="PP",
            start_time=start_time,
            duration=duration,
            int_args=int_args,
            float_args=float_args,
            string_args={},
        )
    dftracer.get_instance().exit_event()  # type: ignore
