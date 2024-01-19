cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)

option(NUFFT_LS_BUILD_PYTHON "Build the pybind11 Python extensions" ON)

# add_compile_options(-Wall -pedantic -Wextra -Werror -Ofast -xHost -funroll-loops -fiopenmp)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
else()
    message(STATUS "No CUDA compiler found; GPU support will be disabled")
endif()

if(NUFFT_LS_BUILD_PYTHON)
  set(PYBIND11_NEWPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(cpu ${CMAKE_CURRENT_LIST_DIR}/src/nufft_ls/cpu.cpp)
  target_include_directories(cpu PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
  target_compile_options(cpu PRIVATE -Wall -pedantic -Wextra -Werror -Ofast -march=native -funroll-loops -fopenmp)
  install(TARGETS cpu LIBRARY DESTINATION .)

  if(CMAKE_CUDA_COMPILER)
    pybind11_add_module(cuda ${CMAKE_CURRENT_LIST_DIR}/src/nufft_ls/cuda.cu)
    target_include_directories(cuda PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
    target_compile_options(cuda PRIVATE -Xptxas -v)
    if (CMAKE_BUILD_TYPE MATCHES Debug)
      target_compile_options(cuda PRIVATE -G -g -lineinfo)
    endif()
    install(TARGETS cuda LIBRARY DESTINATION .)
  endif()
endif()

option(Build_Tests "Build tests" OFF)
if(Build_Tests)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
  enable_testing()
  add_subdirectory(tests)
endif()
