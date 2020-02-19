import os
import shutil
import time
import ctypes as ct
import numpy as np

builderfolder_func = lambda filename: f'build_{filename}'

def build_cpp_project(filename,do_print=False,force=False,clean=True):
    """ build cpp project using CMake """
    
    buildfolder = builderfolder_func(filename) 

    # a. check if build exists
    if os.path.isdir(buildfolder):
        if not force:
            if do_print:
                print(f'{buildfolder} is already build')
            return
        else:
            shutil.rmtree(buildfolder)
            time.sleep(2)
            
    # b. make build
    os.mkdir(buildfolder)
    os.mkdir(buildfolder + '/build/')
    
    # c. write CMakeLists.txt
    CMakeLists_txt = ''
    CMakeLists_txt += 'PROJECT(project)\n'
    CMakeLists_txt += 'cmake_minimum_required(VERSION 2.8)\n'
    CMakeLists_txt += 'set(SuiteSparse_DIR "C:/suitesparse-metis-for-windows/build/SuiteSparse")\n'
    CMakeLists_txt += 'find_package(SuiteSparse CONFIG REQUIRED)\n'
    CMakeLists_txt += 'set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -openmp -Ox")\n'
    CMakeLists_txt += f'add_library({filename} SHARED ../{filename}.cpp)\n'
    CMakeLists_txt += 'include_directories("C:/suitesparse-metis-for-windows/build/install/include/suitesparse")\n'
    CMakeLists_txt += f'target_link_libraries({filename} ${{SuiteSparse_LIBRARIES}})\n'
    
    with open(f'{buildfolder}/CMakeLists.txt', 'w') as txtfile:
        txtfile.write(CMakeLists_txt)
    
    # d. call CMake
    batfile_txt = f'"C:/Program Files/CMake/bin/cmake.exe" -S{buildfolder} -B{buildfolder}/build -G"Visual Studio 15 2017 Win64"'
    
    with open('build.bat', 'w') as batfile:
        batfile.write(batfile_txt)
        
    os.system('build.bat')

    # e. clean
    if clean:
        os.remove('build.bat')

def compile_cpp(filename,do_print=False,force=False,clean=True):
    """ compile cpp files using Visual Studio"""
    
    buildfolder = builderfolder_func(filename) 

    # a. check if build exists
    if os.path.isdir(f'{buildfolder}/build/Release/'):
        if not force:
            if do_print:
                print(f'{filename} already compiled')
            return

    # a. write compile.bat
    batfile_txt = ''
    batfile_txt += 'call "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/vcvarsall.bat" x64\n'
    batfile_txt += f'call msbuild "{os.getcwd()}/{buildfolder}/build/project.sln" /p:Configuration=Release'
    
    with open('compile.bat', 'w') as batfile:
        batfile.write(batfile_txt)
        
    # b. run compile.bat
    os.system('compile.bat')

    # c. clean
    if clean:
        os.remove('compile.bat')

def link(filename,do_print=False):
    """ link to filename with hack for getting openmp to work """
    
    buildfolder = builderfolder_func(filename) 

    # a. load
    cppfile = ct.cdll.LoadLibrary(f'{buildfolder}/build/Release/{filename}.dll')
    
    # b. setup openmp and delink
    cppfile.setup_omp()
    delink(cppfile,do_print=False)
    
    # c. link again
    cppfile = ct.cdll.LoadLibrary(f'{buildfolder}/build/Release/{filename}.dll')
    
    if do_print:
        print('cppfile linked succesfully')
    
    return cppfile

def delink(cppfile,do_print=False):
    """ delinking cppfile is necessary before recompiling 
    (otherwise kernal must be re-started) """
    
    # a. get handle
    handle = cppfile._handle

    # b. delete linking variable
    del cppfile

    # c. free handle
    ct.windll.kernel32.FreeLibrary.argtypes = [ct.wintypes.HMODULE]
    ct.windll.kernel32.FreeLibrary(handle)
    
    if do_print:
        print('cppfile delinked succesfully')        