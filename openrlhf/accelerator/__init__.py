from .interface import Accelerator, AcceleratorEnum, UnspecifiedAccelerator


current_accelerator: Accelerator


is_cuda = False
try:
    import torch

    if torch.cuda.device_count() > 0:
        is_cuda = True
except Exception:
    pass


is_npu = False
try:
    import torch_npu  # noqa: F401

    if torch.npu.device_count() > 0:
        is_npu = True
except Exception:
    pass


if is_cuda:
    from .cuda import CudaAccelerator
    current_accelerator = CudaAccelerator()
elif is_npu:
    from .npu import NpuAccelerator
    current_accelerator = NpuAccelerator()
else:
    current_accelerator = UnspecifiedAccelerator()

__all__ = ["Accelerator", "AcceleratorEnum", "current_platform"]