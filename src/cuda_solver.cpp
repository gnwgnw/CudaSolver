//
// Created by tsv on 05.05.16.
//

#include "cuda_solver.hpp"


CudaSolver::CudaSolver()
		: n(2)
		, x_0(0.0f)
		, x_1(1.0f)
		, t(0.0)
		, t_end(1.0)
		, tau(1.0)
		, threads(1024)
		, blocks(threads)
		, x(n)
		, y(n)
		, done(false)
		, is_x_cached(false)
		, is_y_cached(true)
{
	change_h();
	change_data_size();
	change_grids();

	x_0.connect(boost::bind(&CudaSolver::change_h, this));
	x_1.connect(boost::bind(&CudaSolver::change_h, this));

	h.connect(boost::bind(&CudaSolver::change_n, this));

	n.connect(boost::bind(&CudaSolver::change_h, this));
	n.connect(boost::bind(&CudaSolver::change_grids, this));
	n.connect(boost::bind(&CudaSolver::change_data_size, this));
	n.connect(boost::bind(&CudaSolver::change_xy, this));
	n.connect(boost::bind(&CudaSolver::change_d_y, this));

	alloc_dev_mem();
}

CudaSolver::~CudaSolver()
{
	try {
		free_dev_mem();
	}
	catch (const CudaException& exception) {
		std::cerr << exception.what() << std::endl;
	}
}

bool
CudaSolver::is_done()
{
	return done;
}

ObservableValue<size_t>&
CudaSolver::get_n()
{
	return n;
}

const vector<float>&
CudaSolver::get_x()
{
	if (!is_x_cached) {
		fill_x();
	}

	return x;
}

const vector<float>&
CudaSolver::get_y()
{
	if (!is_y_cached) {
		copy_from_device();
	}

	return y;
}

ObservableValue<float>&
CudaSolver::get_x_0()
{
	return x_0;
}

ObservableValue<float>&
CudaSolver::get_x_1()
{
	return x_1;
}

ObservableValue<float>&
CudaSolver::get_h()
{
	return h;
}

const ObservableValue<double>&
CudaSolver::get_t()
{
	return t;
}

ObservableValue<double>&
CudaSolver::get_t_end()
{
	return t_end;
}

ObservableValue<double>&
CudaSolver::get_tau()
{
	return tau;
}

ObservableValue<bool>&
CudaSolver::get_done()
{
	return done;
}

size_t
CudaSolver::get_threads()
{
	return threads;
}

size_t
CudaSolver::get_blocks()
{
	return blocks;
}

size_t
CudaSolver::get_grids()
{
	return grids;
}

void
CudaSolver::fill_x()
{
	float x_it = x_0;
	float h = this->h;

	x[0] = x_it;
	std::generate(x.begin() + 1, x.end(), [&x_it, h] {
		return x_it += h;
	});

	is_x_cached = true;
}

void
CudaSolver::change_h()
{
	h = (x_1 - x_0) / n;
	is_x_cached = false;
}

void
CudaSolver::change_n()
{
	n = (x_1 - x_0) / h;
}

void
CudaSolver::change_xy()
{
	x.resize(n);
	y.resize(n);
}

void
CudaSolver::change_d_y()
{
	free_dev_mem();
	alloc_dev_mem();
}

void
CudaSolver::change_data_size()
{
	data_size = n * sizeof(float);
}

void
CudaSolver::change_grids()
{
	grids = n / threads + (n % threads ? 1 : 0);
}
