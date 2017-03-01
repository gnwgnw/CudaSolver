//
// Created by tsv on 24.02.17.
//

#ifndef CUDASOLVER_OBSERVABLEVALUE_HPP
#define CUDASOLVER_OBSERVABLEVALUE_HPP

#include <boost/signals2.hpp>


template<class T>
class ObservableValue {
public:
	using signal_t = boost::signals2::signal<void()>;

	explicit ObservableValue(const T& value = T())
			: value(value)
	{
	}

	ObservableValue(const ObservableValue& rhs)
			: ObservableValue(rhs.value)
	{
	}

	virtual
	~ObservableValue()
	{
		signal.disconnect_all_slots();
	}

	const T&
	get() const
	{
		return value;
	}

	void
	set(const T& value)
	{
		if (this->value != value) {
			this->value = value;
			signal();
		}
	}

	void
	connect(const signal_t::slot_type& slot) const
	{
		signal.connect(slot);
	}

	operator T() const
	{
		return value;
	}

	const ObservableValue&
	operator=(const T& value)
	{
		set(value);
		return *this;
	}

	const ObservableValue&
	operator+=(const T& value)
	{
		set(this->value + value);
		return *this;
	}

private:
	T value;
	mutable signal_t signal;
};


#endif //CUDASOLVER_OBSERVABLEVALUE_HPP
