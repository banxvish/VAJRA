"""
KAVACHA AI Voice Defense Engine — Device Selection
====================================================
"""

import torch


def get_device() -> torch.device:
    """Select CUDA GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info() -> str:
    """Return human-readable device information string."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        return f"CUDA — {name} ({mem:.1f} GB)"
    return "CPU"
