import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = [os.path.join(f'pybind_cuda.cpp'), os.path.join(f'embedding.cu'), os.path.join(f'gelu.cu')]

setup(
    name='ByteGLM',
    version='1.0.0',
    ext_modules=[
        CUDAExtension('ByteGLM.lib',
            sources=sources,
            extra_compile_args = {
                'cxx': ['-g', '-O3', '-fopenmp', '-lgomp'],
                'nvcc': ['-O3']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })