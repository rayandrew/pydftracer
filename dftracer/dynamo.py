from dataclasses import dataclass
from typing import List, Optional
import os
from dftracer.logger import dftracer
import torch


DFTRACER_ENABLE_ENV = "DFTRACER_ENABLE"
DFTRACER_ENABLE = True if os.getenv(DFTRACER_ENABLE_ENV, "0") == "1" else False

if DFTRACER_ENABLE:
    try:
        import torch
        from functorch.compile import make_boxed_func

        # Alpha feature from: https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html#custom-backends-after-aotautograd
        from torch._dynamo.backends.common import aot_autograd
    except ImportError:
        raise RuntimeError("DFTracer requires PyTorch to be installed")


# Data structure for trace records
@dataclass
class TraceRecord:
    name: str  # Full function name with layer/subfunction
    cat: str
    timestamp_us: int  # Timestamp in microseconds since epoch
    duration_us: float  # Duration in microseconds
    grad_enabled: bool  # Whether gradients are enabled
    # TODO: add input and output shapes


# function wrapper to enable model tracing using dynamo
class dft_fn:
    __instance = None

    def __init__(self, name: str = "dynamo", enabled: bool = True):
        self._enabled = enabled
        if DFTRACER_ENABLE and self._enabled:
            if self.__instance is not None:
                # We can only have one instance, since it relies on the call_stack to trace, reuse same dft_fn for multiple models
                raise RuntimeError("dft_fn instance already exists")
            self.traces: List[TraceRecord] = []
            self.call_stack = []
            self.df_log: dftracer = dftracer.get_instance()  # type: ignore
            self.name = name
            self.__instance = self

    @classmethod
    def get_instance(cls):
        if not cls.__instance:
            cls.__instance = dft_fn()
        return cls.__instance

    def _reset(self):
        """Reset the tracer state"""
        if DFTRACER_ENABLE and self._enabled:
            self.traces.clear()
            self.call_stack.clear()

    def log_event(self, event: TraceRecord):
        if not (DFTRACER_ENABLE and self._enabled):
            return
        self.df_log.enter_event()
        string_args = {
            "grad_enabled": str(event.grad_enabled),
            # TODO: Add input and output shapes
        }
        # print("Logging: ", event.name)
        self.df_log.log_event(
            name=event.name,
            cat=event.cat,
            start_time=event.timestamp_us,
            duration=event.duration_us,
            string_args=string_args,
        )
        self.df_log.exit_event()

    def compile(self, f_py=None, autograd: bool = True):
        """Used to compile torch.nn.Module and the forward functions
        - Uses PyTorch Dynamo to add nodes before and after each operation
        - The nodes are run before and after each operation -> providing detailed timing information
        """
        if not (DFTRACER_ENABLE and self._enabled):
            # If DFTracer is not enabled, return the function as is
            return f_py

        def _decorator(func):

            def enhanced_trace_wrapper(gm: torch.fx.GraphModule, example, **kwargs):
                """Enhanced custom trace function for Dynamo"""
                # if not (DFTRACER_ENABLE and self._enabled):
                #     return

                # Add enhanced timing instrumentation
                instrumented_gm = add_enhanced_timing_to_graph(gm)
                if autograd:
                    return make_boxed_func(instrumented_gm.forward)
                else:
                    return instrumented_gm.forward

            if autograd:
                return torch.compile(
                    func, backend=aot_autograd(fw_compiler=enhanced_trace_wrapper)
                )
            else:
                return torch.compile(func, backend=enhanced_trace_wrapper)

        return _decorator(f_py) if callable(f_py) else _decorator

    def count_parameters(self, module) -> Optional[int]:
        """Count parameters in a module"""
        if hasattr(module, "parameters"):
            return sum(p.numel() for p in module.parameters())
        return None


def add_enhanced_timing_to_graph(gm):
    """Add enhanced timing nodes to the FX graph"""
    graph = gm.graph

    # Insert timing calls before and after each operation
    nodes_to_instrument = []
    for node in graph.nodes:
        # We do not want to profile placeholder nodes
        if node.op in ["call_module", "call_function", "call_method"]:
            nodes_to_instrument.append(node)

    for node in reversed(nodes_to_instrument):
        op_name = create_detailed_op_name(node, gm)

        with graph.inserting_before(node):
            graph.call_function(
                record_time_start,
                args=(op_name, node.op, str(node.target)),
            )

        with graph.inserting_after(node):
            graph.call_function(record_time_end, args=(op_name, node))

    gm.recompile()
    # Print graph
    # gm.print_readable()
    return gm


def record_time_start(op_name: str, op_type: str, target: str):
    """Record start time for an operation"""
    self = dft_fn.get_instance()
    timestamp_us = self.df_log.get_time()

    start_info = {
        "op_name": op_name,
        "op_type": op_type,
        "target": target,
        "timestamp_us": timestamp_us,
        "grad_enabled": torch.is_grad_enabled(),
    }

    self.call_stack.append(start_info)
    # print("time elapsed instart fn: ", tracer.df_log.get_time() - timestamp_us)
    return None


def record_time_end(op_name: str, node):
    """Record end time for an operation"""
    self = dft_fn.get_instance()
    if not self.call_stack:
        return None

    # The start_info is at the top of the call stack because of the graph structure
    start_info = self.call_stack.pop()
    end_timestamp_us = self.df_log.get_time()

    # Calculate duration
    duration_us = end_timestamp_us - start_info["timestamp_us"]

    # Determine if this is forward or backward pass
    # Create trace record
    record = TraceRecord(
        name=op_name,
        cat=self.name,
        timestamp_us=start_info["timestamp_us"],
        duration_us=duration_us,
        grad_enabled=start_info["grad_enabled"],
    )

    self.log_event(record)

    return None


def create_detailed_op_name(node, gm) -> str:
    """Create detailed operation name with layer/subfunction info"""
    base_name = str(node.target).replace(".", "_")

    if node.op == "call_module":
        module_path = str(node.target)
        module = gm.get_submodule(node.target)
        module_type = type(module).__name__
        return f"{module_path}.{module_type}_{base_name}"
    elif node.op == "call_function":
        func_name = getattr(node.target, "__name__", str(node.target))
        return f"function.{func_name}"
    elif node.op == "call_method":
        return f"method.{base_name}"
    else:
        return f"{node.op}.{base_name}"
