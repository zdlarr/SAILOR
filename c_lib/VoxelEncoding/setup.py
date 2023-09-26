from setuptools import setup, find_packages
import os, shutil
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension('VoxelEncoding', 
        sources = [
            './src/fusion_kernel.cu',
            './src/octree_kernel.cu',
            './src/intersection_kernel.cu',
            './src/sampling_kernel.cu',
            './src/binding.cpp',
            './src/undistort.cu',
            './src/volume_rendering.cu',
            './src/freq_encoding.cu',
            './src/sh_encoding.cu',
            './src/depth_normalizer.cu',
            './src/udf.cu',
            './src/tensor_where.cu',
            './src/depth2color.cu'
        ],
        include_dirs = ['/usr/local/include'],
        libraries = ['opencv_core', 'opencv_highgui', 'opencv_imgcodecs'],
        extra_compile_args= {'nvcc': ["-Xptxas", "-v"]},
        library_dirs = ['/usr/local/lib']
    )
]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'opencv-python']

setup(
    name='voxel_encoding',
    description='Integrate the TSDF Volume, Octree structure, Voxel&Octree intersection, sampling and other properties',
    version='0.1',
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)

# move the lib to target dirs;
for file in os.listdir('./build/lib.linux-x86_64-3.8'):
    src = os.path.join('./build/lib.linux-x86_64-3.8', file)
    dst = os.path.join('./dist', file)
    shutil.move(src, dst)