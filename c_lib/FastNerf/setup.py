from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import shutil, os

ext_modules = [
    CUDAExtension('InferNerf', 
        sources = [
            './src/binding.cpp',
            './src/infer_mlp.cu'
        ],
        include_dirs = ['/usr/local/include'],
        libraries = ['opencv_core', 'opencv_highgui', 'opencv_imgcodecs'],
        extra_compile_args= {'nvcc': ["-Xptxas", "-v"]},
        library_dirs = ['/usr/local/lib']
    )
]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'opencv-python']


setup(
    name='InferNerf',
    description='Integrate the Nerf MLP framework',
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