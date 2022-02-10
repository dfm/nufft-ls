cmake_minimum_required(VERSION 3.12)
project(jax_finufft LANGUAGES C CXX)
message(STATUS "Using CMake version: " ${CMAKE_VERSION})

# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)
# find_package(FFTW REQUIRED COMPONENTS FLOAT_LIB DOUBLE_LIB)
# link_libraries(${FFTW_FLOAT_LIB} ${FFTW_DOUBLE_LIB})

add_compile_options(-Wall -pedantic -Wextra -Werror -O3 -funroll-loops -fopenmp)
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

# set(FINUFFT_INCLUDE_DIRS
#     ${CMAKE_CURRENT_LIST_DIR}/lib
#     ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/include
#     ${FFTW_INCLUDE_DIRS})

# message(STATUS "FINUFFT include dirs: " "${FINUFFT_INCLUDE_DIRS}")

# add_library(finufft STATIC
#     ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/spreadinterp.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/utils.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/finufft.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/simpleinterfaces.cpp)
# target_include_directories(finufft PRIVATE ${FINUFFT_INCLUDE_DIRS})

# add_library(finufft_32 STATIC
#     ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/spreadinterp.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/utils.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/finufft.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/simpleinterfaces.cpp)
# target_compile_definitions(finufft_32 PUBLIC SINGLE)
# target_include_directories(finufft_32 PRIVATE ${FINUFFT_INCLUDE_DIRS})

# pybind11_add_module(jax_finufft
#     ${CMAKE_CURRENT_LIST_DIR}/lib/jax_finufft.cc
#     ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/utils_precindep.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/contrib/legendre_rule_fast.c)
# target_link_libraries(jax_finufft PRIVATE finufft finufft_32)
# target_include_directories(jax_finufft PRIVATE ${FINUFFT_INCLUDE_DIRS})
# install(TARGETS jax_finufft DESTINATION .)

option(Build_Tests "Build tests" ON)
if(Build_Tests)
  enable_testing()
  add_subdirectory(tests)
endif()
