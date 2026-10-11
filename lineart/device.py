"""Pick the fastest available torch device: CUDA, Apple Silicon (MPS), or CPU."""
import os
import torch


def pick():
    """Return (device, dtype). Override with LINEART_DEVICE=cpu|mps|cuda."""
    want = os.environ.get("LINEART_DEVICE")
    if want == "cuda" or (not want and torch.cuda.is_available()):
        return "cuda", torch.float16
    if want == "mps" or (not want and torch.backends.mps.is_available()):
        return "mps", torch.float16
    torch.set_num_threads(os.cpu_count() or 4)
    return "cpu", torch.float32


def prepare(pipe, device):
    """Move a diffusers pipeline to the device and apply memory savers where they help."""
    pipe = pipe.to(device)
    if device == "mps":
        # diffusers' MPS docs recommend this below 64 GB of unified memory
        pipe.enable_attention_slicing()
    return pipe
