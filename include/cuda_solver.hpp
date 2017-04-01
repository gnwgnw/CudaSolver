//
// Created by tsv on 29.04.16.
//

#ifndef CUDASOLVER_CUDA_SOLVER_HPP
#define CUDASOLVER_CUDA_SOLVER_HPP


#include "utils.hpp"
#include "observable_value.hpp"

#include <iostream>
#include <boost/container/vector.hpp>


template<class T>
using vector = boost::container::vector<T>;


class CudaSolver {
public:
	CudaSolver();

	virtual
	~CudaSolver();

	void
	step();

	virtual
	bool
	is_done();

	ObservableValue<size_t>&
	get_n();

	ObservableValue<float>&
	get_x_0();

	ObservableValue<float>&
	get_x_1();

	ObservableValue<float>&
	get_h();

	const ObservableValue<double>&
	get_t();

	ObservableValue<double>&
	get_t_end();

	ObservableValue<double>&
	get_tau();

	ObservableValue<bool>&
	get_done();

	const vector<float>&
	get_x();

	const vector<float>&
	get_y();

	void
	set_y(const vector<float>& y);

protected:
	float* d_y_in;
	float* d_y_out;

	size_t
	get_threads();

	size_t
	get_blocks();

	size_t
	get_grids();


private:
	ObservableValue<size_t> n;

	ObservableValue<float> x_0;
	ObservableValue<float> x_1;
	ObservableValue<float> h;

	ObservableValue<double> t;
	ObservableValue<double> t_end;
	ObservableValue<double> tau;

	size_t data_size;

	size_t threads;
	size_t blocks;
	size_t grids;

	vector<float> x;
	vector<float> y;

	ObservableValue<bool> done;
	bool is_x_cached;
	bool is_y_cached;

	virtual
	void
	next_step() = 0;

	void
	fill_x();

	void
	copy_from_device();

	void
	alloc_dev_mem();

	void
	free_dev_mem();

	void
	change_h();

	void
	change_n();

	void
	change_xy();

	void
	change_d_y();

	void
	change_data_size();

	void
	change_grids();
};


#endif //CUDASOLVER_CUDA_SOLVER_HPP
