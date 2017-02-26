//
// Created by tsv on 22.01.17.
//

#ifndef CUDASOLVER_UTILS_HPP
#define CUDASOLVER_UTILS_HPP


#include <exception>
#include <string>


class CudaException : public std::exception {
public:
	explicit CudaException(const char* message)
			: msg(message)
	{
	}

	explicit CudaException(const std::string message)
			: msg(message)
	{
	}

	~CudaException() = default;

	const char*
	what() const noexcept override
	{
		return msg.c_str();
	}

protected:
	std::string msg;
};


#endif //CUDASOLVER_UTILS_HPP
