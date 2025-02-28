import enum
from typing import Tuple

import torch


class AcceleratorEnum(enum.Enum):
    CUDA = enum.auto()
    NPU = enum.auto()
    UNSPECIFIED = enum.auto()


class Accelerator:
    _enum: AcceleratorEnum

    def is_cuda(self) -> bool:
        return self._enum == AcceleratorEnum.CUDA
    
    def is_npu(self) -> bool:
        return self._enum == AcceleratorEnum.NPU
    
    # Device APIs
    @staticmethod
    def device_name(device_index=None):
        raise NotImplementedError

    @staticmethod
    def set_device(device_index=None):
        raise NotImplementedError

    @staticmethod
    def current_device():
        raise NotImplementedError
    
    @staticmethod
    def current_device_name():
        raise NotImplementedError
    
    @staticmethod
    def device_count():
        raise NotImplementedError
    
    # RNG APIs
    def manual_seed(seed):
        raise NotImplementedError
    
    # Momory management
    @staticmethod
    def empty_cache():
        raise NotImplementedError
    
    @staticmethod
    def synchronize():
        raise NotImplementedError
    

class UnspecifiedAccelerator(Accelerator):
    _enum = AcceleratorEnum.UNSPECIFIED
