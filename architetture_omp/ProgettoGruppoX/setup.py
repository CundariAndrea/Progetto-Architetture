from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import numpy as np
import glob
import os

gruppo='gruppoX'

class CustomBuildExt(build_ext):
    def run(self):
        # Compila file NASM prima di build C
        for arch in ['32', '64', '64omp']:
            folder = f"src/{arch}"
            nasm_files = glob.glob(os.path.join(folder, "*.nasm"))
            for nasm_file in nasm_files:
                subprocess.run([
                    'nasm',
                    '-f', 'elf64',
                    '-DPIC',
                    '-I', folder,
                    nasm_file
                ], check=True)

        # Aggiunge i file .o dinamicamente
        for ext in self.extensions:
            if '32' in ext.name:
                obj_files = glob.glob('src/32/*.o')
                ext.extra_objects = obj_files
            elif '64omp' in ext.name:
                obj_files = glob.glob('src/64omp/*.o')
                ext.extra_objects = obj_files
            elif '64' in ext.name:
                obj_files = glob.glob('src/64/*.o')
                ext.extra_objects = obj_files
        super().run()

module32 = Extension(
    f"{gruppo}.quantpivot32._quantpivot32",  # Nome completo del modulo
    sources=['src/32/quantpivot32_py.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=['-O0', '-msse', '-fPIC'],
    extra_link_args=['-z', 'noexecstack']
)

module64 = Extension(
    f"{gruppo}.quantpivot64._quantpivot64",  # Nome completo del modulo
    sources=['src/64/quantpivot64_py.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=['-O0', '-msse', '-mavx', '-fPIC'],
    extra_link_args=['-z', 'noexecstack']
)

module64omp = Extension(
    f"{gruppo}.quantpivot64omp._quantpivot64omp",  # Nome completo del modulo
    sources=['src/64omp/quantpivot64omp_py.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=['-O0', '-msse', '-mavx', '-fPIC', '-fopenmp'],
    extra_link_args=['-z', 'noexecstack', '-fopenmp']
)

setup(
    name=gruppo,
    version='1.0',
    author="LISTA COMPONENTI GRUPPO",
    packages=find_packages(),  # Trova automaticamente i pacchetti
    ext_modules=[module32, module64, module64omp],
    cmdclass={'build_ext': CustomBuildExt},
    install_requires=['numpy'],
    zip_safe=False             # Non eseguibile da zip senza scompattarlo
)
