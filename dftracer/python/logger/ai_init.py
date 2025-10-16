from dftracer.logger.ai_common import *

ai = AI()
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
