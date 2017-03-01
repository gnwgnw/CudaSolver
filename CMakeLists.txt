cmake_minimum_required(VERSION 3.6)
project(CudaSolver)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)

find_package(CUDA REQUIRED)
find_package(Boost REQUIRED)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})


set(CUDA_SOURCE_FILES
		src/cuda_solver.cu
		include/cuda_utils.cuh)

cuda_add_library(Solver SHARED ${CUDA_SOURCE_FILES})


set(SOURCE_FILES
		include/cuda_solver.hpp
		src/cuda_solver.cpp
		include/utils.hpp
		include/observable_value.hpp)

add_library(CudaSolver SHARED ${SOURCE_FILES})
target_link_libraries(CudaSolver Solver ${CUDA_LIBRARIES} ${Boost_LIBRARIES})
