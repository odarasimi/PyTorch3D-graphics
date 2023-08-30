from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_cpp',
    ext_modules=[
        CUDAExtension('cuda_cpp', ['interpolation.cpp', 'interpolation_optim.cu']) 
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
