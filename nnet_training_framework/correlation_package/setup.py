#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++14']

nvcc_args = [
    '-gencode', 'arch=compute_61,code=sm_61',
    '-ccbin', '/usr/bin/gcc'
]

setup(
    name='correlation_pkg',
    ext_modules=[
        CUDAExtension('correlation_pkg', [
            'correlation_cuda.cpp',
            'correlation_cuda_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args, 'cuda-path': ['/usr/local/cuda-10.2']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)