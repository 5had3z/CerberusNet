#!/usr/bin/env python3
"""
Change the NVCC args to the correct SM you have and install with the command
> python3 setup.py install
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CXX_ARGS = ['-std=c++14', '-Ofast']

NVCC_ARGS = [
    '-gencode', 'arch=compute_60,code=sm_60',
    '-ccbin', '/usr/bin/gcc'
]

setup(
    name='correlation_pkg',
    ext_modules=[
        CUDAExtension(
            'correlation_pkg',
            ['correlation_cuda.cpp', 'correlation_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': CXX_ARGS, 'nvcc': NVCC_ARGS,
                'cuda-path': ['/usr/local/cuda-10.1']
                }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
