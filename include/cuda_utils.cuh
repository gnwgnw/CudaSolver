#ifndef CUDASOLVER_CUDA_UTILS_CUH
#define CUDASOLVER_CUDA_UTILS_CUH


#include "utils.hpp"

#include <iostream>

#include <cuda_runtime_api.h>


#define cuda_check_error(ans) { cuda_assert((ans), __FILE__, __LINE__); }


inline void
cuda_assert(cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess) {
		auto msg = cudaGetErrorString(code);
		std::cerr << "GPU assert: " << msg << " " << file << " " << line << std::endl;

		throw CudaException(msg);
	}
}


#endif
