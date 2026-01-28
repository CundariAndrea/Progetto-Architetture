import os
import glob
import subprocess
import sys
import numpy as np
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

gruppo = 'gruppo23'

HELPER_DIR = os.path.join('src', 'helper')
helper_sources = glob.glob(os.path.join(HELPER_DIR, '*.c'))

class CustomBuildExt(build_ext):
    def run(self):
        try:
            print("\n--- Compilazione Eseguibile 32 bit (tramite Makefile) ---")
            
            subprocess.check_call(['make', '-C', 'src/32', 'build'])
            print("--- Eseguibile 32 bit creato con successo ---\n")
        except subprocess.CalledProcessError:
            print("ATTENZIONE: Compilazione 32-bit fallita.")
            print("Assicurati di aver installato 'gcc-multilib'.")
            print("Saltiamo questo passaggio e proseguiamo con la 64 bit.\n")
        except FileNotFoundError:
             print("ATTENZIONE: Impossibile trovare 'make'. Assicurati che sia installato.\n")

        for arch in ['64', '64omp']:
            folder = os.path.join("src", arch)
            nasm_files = glob.glob(os.path.join(folder, "*.nasm"))
            
            nasm_fmt = 'elf64'
            
            for nasm_file in nasm_files:
                obj_file = nasm_file.replace('.nasm', '.o')
                print(f"--- Compilazione NASM ({arch}): {nasm_file} -> {obj_file} ---")
                
                cmd = [
                    'nasm', 
                    '-f', nasm_fmt, 
                    '-DPIC', 
                    '-I', folder + os.path.sep, 
                    nasm_file, 
                    '-o', obj_file
                ]
                subprocess.check_call(cmd)
        for ext in self.extensions:
            if 'quantpivot64omp' in ext.name:
                ext.extra_objects = glob.glob('src/64omp/*.o')
            elif 'quantpivot64' in ext.name:
                ext.extra_objects = glob.glob('src/64/*.o')
        
        super().run()


module64 = Extension(
    f"{gruppo}.quantpivot64._quantpivot64",
    sources=['src/64/quantpivot64_py.c', 'src/64/quantpivot64.c'] + helper_sources,
    include_dirs=[np.get_include(), 'src/64', HELPER_DIR],
    extra_compile_args=['-O0', '-msse', '-mavx', '-fPIC', '-DUSE_ASM_IMPL'],
    extra_link_args=['-z', 'noexecstack']
)


module64omp = Extension(
    f"{gruppo}.quantpivot64omp._quantpivot64omp",
    sources=['src/64omp/quantpivot64omp_py.c', 'src/64omp/quantpivot64omp.c'] + helper_sources,
    include_dirs=[np.get_include(), 'src/64omp', HELPER_DIR],
    extra_compile_args=['-O0', '-msse', '-mavx', '-fPIC', '-fopenmp', '-DUSE_ASM_IMPL'],
    extra_link_args=['-z', 'noexecstack', '-fopenmp']
)

setup(
    name=gruppo,
    version='1.0',
    author="GRUPPO 23",
    packages=find_packages(),
    ext_modules=[module64, module64omp],
    cmdclass={'build_ext': CustomBuildExt},
    install_requires=['numpy'],
    zip_safe=False
)