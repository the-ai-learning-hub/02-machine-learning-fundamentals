"""CPU vs. GPU matrix-multiplication benchmark (PyTorch).

GPU calls are *asynchronous*: ``a @ b`` returns before the GPU finishes. To measure real time we
synchronise before reading the clock, warm up first (the first CUDA call pays one-off start-up
costs) and report the median of several repetitions.
"""

from __future__ import annotations

import platform
import statistics
import time


def available_devices() -> list[str]:
    """``["cpu"]`` plus ``"cuda"`` and/or ``"mps"`` when available."""
    import torch

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def device_name(device: str) -> str:
    """Human-readable hardware name for ``device``."""
    import torch

    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return "Apple GPU (MPS)"
    return f"{platform.processor() or platform.machine()} · {torch.get_num_threads()} threads"


def _sync(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_matmul(n: int, device: str = "cpu", repeats: int = 10, warmup: int = 3,
                dtype: str = "float32", seed: int = 0) -> dict[str, float]:
    """Median time (ms) of an ``n×n @ n×n`` product on ``device`` and the achieved GFLOP/s.

    An n×n product needs ≈ 2·n³ floating-point operations.
    """
    import torch

    torch.manual_seed(seed)
    dt = getattr(torch, dtype)
    a = torch.randn(n, n, dtype=dt, device=device)
    b = torch.randn(n, n, dtype=dt, device=device)
    for _ in range(warmup):
        a @ b
    _sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        a @ b
        _sync(device)
        times.append(time.perf_counter() - start)
    ms = statistics.median(times) * 1000
    return {"device": device, "n": n, "ms": ms, "gflops": 2 * n**3 / (ms / 1000) / 1e9}


def time_transfer(n: int, device: str = "cuda", repeats: int = 10) -> float:
    """Median time (ms) to copy an ``n×n`` float32 matrix from the CPU to ``device`` and back."""
    import torch

    x = torch.randn(n, n)
    x.to(device)
    _sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        y = x.to(device)
        y.cpu()
        _sync(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1000
