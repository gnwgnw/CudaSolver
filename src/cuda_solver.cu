#include "../include/cuda_solver.hpp"
#include "../include/cuda_utils.cuh"


void
CudaSolver::step()
{
	if (!is_done()) {
		next_step();
		cuda_check_error(cudaGetLastError());

		std::swap(d_y_in, d_y_out);

		t += tau;
		if (t > t_end) {
			done = true;
		}

		if (is_y_cached) {
			is_y_cached = false;
		}
	}
}

void
CudaSolver::alloc_dev_mem()
{
	cuda_check_error(cudaMalloc((void**) &d_y_in, data_size));
	cuda_check_error(cudaMalloc((void**) &d_y_out, data_size));
	cuda_check_error(cudaMemcpy(d_y_in, y.data(), data_size, cudaMemcpyHostToDevice));
	cuda_check_error(cudaMemcpy(d_y_out, y.data(), data_size, cudaMemcpyHostToDevice));
}

void
CudaSolver::free_dev_mem()
{
	cuda_check_error(cudaFree(d_y_in));
	cuda_check_error(cudaFree(d_y_out));
	d_y_in = nullptr;
	d_y_out = nullptr;
}

void
CudaSolver::copy_from_device()
{
	cuda_check_error(cudaMemcpy(y.data(), d_y_in, data_size, cudaMemcpyDeviceToHost));
	is_y_cached = true;
}
