cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

option(NUFFT_LS_BUILD_PYTHON "Build the pybind11 Python extensions" ON)

add_compile_options(-Wall -pedantic -Wextra -Werror -Ofast -march=native -funroll-loops -fopenmp)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

if(NUFFT_LS_BUILD_PYTHON)
  set(PYBIND11_NEWPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(cpu ${CMAKE_CURRENT_LIST_DIR}/src/nufft_ls/cpu.cpp)

  install(TARGETS cpu LIBRARY DESTINATION .)
endif()

option(Build_Tests "Build tests" ON)
if(Build_Tests)
  enable_testing()
  add_subdirectory(tests)
endif()
