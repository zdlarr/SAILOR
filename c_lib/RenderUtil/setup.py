
import os, shutil
from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension('RenderUtils', [
        './src/render_utils.cpp',
        './src/render_utils_kernel.cu',
    ])
]

INSTALL_REQUIREMENTS = ['numpy', 'torch']

setup(
    name='RenderMesh',
    description='Render from a given mesh.',
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