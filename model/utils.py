import torch

def get_device():
    if torch.cuda.is_available():       # NVIDIA / ROCm
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel GPU
        return torch.device("xpu")
    else:
        return torch.device("cpu")