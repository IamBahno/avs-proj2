set(CMAKE_C_COMPILER "/apps/all/intel-compilers/2024.2.0/compiler/2024.2/bin/icx")
set(CMAKE_C_COMPILER_ARG1 "")
set(CMAKE_C_COMPILER_ID "IntelLLVM")
set(CMAKE_C_COMPILER_VERSION "2024.2.0")
set(CMAKE_C_COMPILER_VERSION_INTERNAL "")
set(CMAKE_C_COMPILER_WRAPPER "")
set(CMAKE_C_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_C_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_C_COMPILE_FEATURES "c_std_90;c_function_prototypes;c_std_99;c_restrict;c_variadic_macros;c_std_11;c_static_assert;c_std_17;c_std_23")
set(CMAKE_C90_COMPILE_FEATURES "c_std_90;c_function_prototypes")
set(CMAKE_C99_COMPILE_FEATURES "c_std_99;c_restrict;c_variadic_macros")
set(CMAKE_C11_COMPILE_FEATURES "c_std_11;c_static_assert")
set(CMAKE_C17_COMPILE_FEATURES "c_std_17")
set(CMAKE_C23_COMPILE_FEATURES "c_std_23")

set(CMAKE_C_PLATFORM_ID "Linux")
set(CMAKE_C_SIMULATE_ID "GNU")
set(CMAKE_C_COMPILER_FRONTEND_VARIANT "GNU")
set(CMAKE_C_SIMULATE_VERSION "4.2.1")




set(CMAKE_AR "/apps/all/binutils/2.42-GCCcore-13.3.0/bin/ar")
set(CMAKE_C_COMPILER_AR "CMAKE_C_COMPILER_AR-NOTFOUND")
set(CMAKE_RANLIB "/apps/all/binutils/2.42-GCCcore-13.3.0/bin/ranlib")
set(CMAKE_C_COMPILER_RANLIB "CMAKE_C_COMPILER_RANLIB-NOTFOUND")
set(CMAKE_LINKER "/apps/all/binutils/2.42-GCCcore-13.3.0/bin/ld")
set(CMAKE_MT "")
set(CMAKE_TAPI "CMAKE_TAPI-NOTFOUND")
set(CMAKE_COMPILER_IS_GNUCC )
set(CMAKE_C_COMPILER_LOADED 1)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_C_ABI_COMPILED TRUE)

set(CMAKE_C_COMPILER_ENV_VAR "CC")

set(CMAKE_C_COMPILER_ID_RUN 1)
set(CMAKE_C_SOURCE_FILE_EXTENSIONS c;m)
set(CMAKE_C_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_C_LINKER_PREFERENCE 10)
set(CMAKE_C_LINKER_DEPFILE_SUPPORTED )

# Save compiler ABI information.
set(CMAKE_C_SIZEOF_DATA_PTR "8")
set(CMAKE_C_COMPILER_ABI "ELF")
set(CMAKE_C_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_C_LIBRARY_ARCHITECTURE "x86_64-unknown-linux-gnu")

if(CMAKE_C_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_C_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_C_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_C_COMPILER_ABI}")
endif()

if(CMAKE_C_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-unknown-linux-gnu")
endif()

set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_C_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_C_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "/apps/all/intel-compilers/2024.2.0/tbb/2021.13/include;/apps/all/binutils/2.42-GCCcore-13.3.0/include;/apps/all/zlib/1.3.1-GCCcore-13.3.0/include;/apps/all/libarchive/3.7.2-GCCcore-13.2.0/include;/apps/all/XZ/5.4.4-GCCcore-13.2.0/include;/apps/all/cURL/8.3.0-GCCcore-13.2.0/include;/apps/all/bzip2/1.0.8-GCCcore-13.2.0/include;/apps/all/ncurses/6.4-GCCcore-13.2.0/include;/apps/all/Qhull/2020.2-GCCcore-11.3.0/include;/apps/all/LibTIFF/4.3.0-GCCcore-11.3.0/include;/apps/all/libdeflate/1.10-GCCcore-11.3.0/include;/apps/all/zstd/1.5.2-GCCcore-11.3.0/include;/apps/all/lz4/1.9.3-GCCcore-11.3.0/include;/apps/all/jbigkit/2.1-GCCcore-11.3.0/include;/apps/all/libjpeg-turbo/2.1.3-GCCcore-11.3.0/include;/apps/all/Tk/8.6.12-GCCcore-11.3.0/include;/apps/all/X11/20220504-GCCcore-11.3.0/include;/apps/all/fontconfig/2.14.0-GCCcore-11.3.0/include;/apps/all/util-linux/2.38-GCCcore-11.3.0/include;/apps/all/expat/2.4.8-GCCcore-11.3.0/include;/apps/all/freetype/2.12.1-GCCcore-11.3.0/include/freetype2;/apps/all/Brotli/1.0.9-GCCcore-11.3.0/include;/apps/all/libpng/1.6.37-GCCcore-11.3.0/include;/apps/all/SciPy-bundle/2022.05-foss-2022a/lib/python3.10/site-packages/numpy/core/include;/apps/all/pybind11/2.9.2-GCCcore-11.3.0/include;/apps/all/Python/3.10.4-GCCcore-11.3.0/include;/apps/all/libffi/3.4.2-GCCcore-11.3.0/include;/apps/all/GMP/6.2.1-GCCcore-11.3.0/include;/apps/all/SQLite/3.38.3-GCCcore-11.3.0/include;/apps/all/Tcl/8.6.12-GCCcore-11.3.0/include;/apps/all/libreadline/8.1.2-GCCcore-11.3.0/include;/apps/all/FFTW.MPI/3.3.10-gompi-2022a/include;/apps/all/FFTW/3.3.10-GCC-11.3.0/include;/apps/all/FlexiBLAS/3.2.0-GCC-11.3.0/include;/apps/all/OpenBLAS/0.3.20-GCC-11.3.0/include;/apps/all/OpenMPI/4.1.4-GCC-11.3.0/include;/apps/all/UCC/1.0.0-GCCcore-11.3.0/include;/apps/all/PMIx/4.1.2-GCCcore-11.3.0/include;/apps/all/libfabric/1.15.1-GCCcore-11.3.0/include;/apps/all/UCX/1.12.1-GCCcore-11.3.0/include;/apps/all/libevent/2.1.12-GCCcore-11.3.0/include;/apps/all/OpenSSL/1.1/include;/apps/all/hwloc/2.7.1-GCCcore-11.3.0/include;/apps/all/libpciaccess/0.16-GCCcore-11.3.0/include;/apps/all/libxml2/2.9.13-GCCcore-11.3.0/include/libxml2;/apps/all/libxml2/2.9.13-GCCcore-11.3.0/include;/apps/all/numactl/2.0.14-GCCcore-11.3.0/include;/apps/all/intel-compilers/2024.2.0/compiler/2024.2/opt/compiler/include;/apps/all/intel-compilers/2024.2.0/compiler/2024.2/lib/clang/19/include;/usr/local/include;/usr/include")
set(CMAKE_C_IMPLICIT_LINK_LIBRARIES "svml;irng;imf;m;gcc;gcc_s;irc;dl;gcc;gcc_s;c;gcc;gcc_s;irc_s")
set(CMAKE_C_IMPLICIT_LINK_DIRECTORIES "/apps/all/intel-compilers/2024.2.0/compiler/2024.2/lib;/apps/all/intel-compilers/2024.2.0/compiler/2024.2/lib/clang/19/lib/x86_64-unknown-linux-gnu;/apps/all/GCCcore/13.3.0/lib/gcc/x86_64-pc-linux-gnu/13.3.0;/apps/all/GCCcore/13.3.0/lib64;/lib64;/usr/lib64;/apps/all/GCCcore/13.3.0/lib;/apps/all/intel-compilers/2024.2.0/compiler/2024.2/opt/compiler/lib;/lib;/usr/lib;/apps/all/intel-compilers/2024.2.0/tbb/2021.13/lib/intel64/gcc4.8;/apps/all/binutils/2.42-GCCcore-13.3.0/lib;/apps/all/zlib/1.3.1-GCCcore-13.3.0/lib;/apps/all/libarchive/3.7.2-GCCcore-13.2.0/lib;/apps/all/XZ/5.4.4-GCCcore-13.2.0/lib;/apps/all/cURL/8.3.0-GCCcore-13.2.0/lib;/apps/all/bzip2/1.0.8-GCCcore-13.2.0/lib;/apps/all/ncurses/6.4-GCCcore-13.2.0/lib;/apps/all/matplotlib/3.5.2-foss-2022a/lib;/apps/all/Qhull/2020.2-GCCcore-11.3.0/lib;/apps/all/LibTIFF/4.3.0-GCCcore-11.3.0/lib;/apps/all/libdeflate/1.10-GCCcore-11.3.0/lib;/apps/all/zstd/1.5.2-GCCcore-11.3.0/lib;/apps/all/lz4/1.9.3-GCCcore-11.3.0/lib;/apps/all/jbigkit/2.1-GCCcore-11.3.0/lib;/apps/all/libjpeg-turbo/2.1.3-GCCcore-11.3.0/lib;/apps/all/Tkinter/3.10.4-GCCcore-11.3.0/lib;/apps/all/Tk/8.6.12-GCCcore-11.3.0/lib;/apps/all/X11/20220504-GCCcore-11.3.0/lib;/apps/all/fontconfig/2.14.0-GCCcore-11.3.0/lib;/apps/all/util-linux/2.38-GCCcore-11.3.0/lib;/apps/all/expat/2.4.8-GCCcore-11.3.0/lib;/apps/all/freetype/2.12.1-GCCcore-11.3.0/lib;/apps/all/Brotli/1.0.9-GCCcore-11.3.0/lib;/apps/all/libpng/1.6.37-GCCcore-11.3.0/lib;/apps/all/SciPy-bundle/2022.05-foss-2022a/lib/python3.10/site-packages/numpy/core/lib;/apps/all/SciPy-bundle/2022.05-foss-2022a/lib;/apps/all/Python/3.10.4-GCCcore-11.3.0/lib;/apps/all/libffi/3.4.2-GCCcore-11.3.0/lib64;/apps/all/libffi/3.4.2-GCCcore-11.3.0/lib;/apps/all/GMP/6.2.1-GCCcore-11.3.0/lib;/apps/all/SQLite/3.38.3-GCCcore-11.3.0/lib;/apps/all/Tcl/8.6.12-GCCcore-11.3.0/lib;/apps/all/libreadline/8.1.2-GCCcore-11.3.0/lib;/apps/all/ScaLAPACK/2.2.0-gompi-2022a-fb/lib;/apps/all/FFTW.MPI/3.3.10-gompi-2022a/lib;/apps/all/FFTW/3.3.10-GCC-11.3.0/lib;/apps/all/FlexiBLAS/3.2.0-GCC-11.3.0/lib;/apps/all/OpenBLAS/0.3.20-GCC-11.3.0/lib;/apps/all/OpenMPI/4.1.4-GCC-11.3.0/lib;/apps/all/UCC/1.0.0-GCCcore-11.3.0/lib;/apps/all/PMIx/4.1.2-GCCcore-11.3.0/lib;/apps/all/libfabric/1.15.1-GCCcore-11.3.0/lib;/apps/all/UCX/1.12.1-GCCcore-11.3.0/lib;/apps/all/libevent/2.1.12-GCCcore-11.3.0/lib;/apps/all/OpenSSL/1.1/lib;/apps/all/hwloc/2.7.1-GCCcore-11.3.0/lib;/apps/all/libpciaccess/0.16-GCCcore-11.3.0/lib;/apps/all/libxml2/2.9.13-GCCcore-11.3.0/lib;/apps/all/numactl/2.0.14-GCCcore-11.3.0/lib")
set(CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
