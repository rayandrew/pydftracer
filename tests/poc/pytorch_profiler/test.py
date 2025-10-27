import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from dftracer.python import dftracer
from dftracer.python.torch import trace_handler
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# 2. End-to-End Execution with Profiler
if __name__ == "__main__":
    df_logger = dftracer.initialize_log("test.pfw", None, -1)
    print("Logger initialized")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data setup
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2
    )

    # Model, loss, and optimizer setup
    model = models.resnet18(weights=None).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # 3. Profiler schedule and loop control
    WAIT_STEPS = 1
    WARMUP_STEPS = 1
    ACTIVE_STEPS = 3
    REPEAT_CYCLES = 1

    # Calculate the total number of steps to run the training loop
    total_steps = (WAIT_STEPS + WARMUP_STEPS + ACTIVE_STEPS) * REPEAT_CYCLES

    profiler_schedule = schedule(
        wait=WAIT_STEPS,
        warmup=WARMUP_STEPS,
        active=ACTIVE_STEPS,
        repeat=REPEAT_CYCLES,
    )

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA if device.type == "cuda" else ProfilerActivity.CPU,
        ],
        schedule=profiler_schedule,
        on_trace_ready=trace_handler,
        profile_memory=True,
        with_stack=True,
    ) as p:
        # Loop for a specific number of batches, corresponding to the profiler schedule
        for step, (inputs, labels) in enumerate(train_dataloader):
            if step >= total_steps:
                break

            print(f"Step {step + 1}/{total_steps}")
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with record_function("model_forward"):
                outputs = model(inputs)

            with record_function("loss_calculation"):
                loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            p.step()

    print("Profiling session complete.")
    df_logger.finalize()
