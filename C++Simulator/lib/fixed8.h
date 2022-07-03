#pragma once
#include <cstdint>
#include "qLib8.h"
#include <iostream>

class fixed8
{
private:
	uint16_t _bin;
	int _m;
	int _n;

public:
	//constructor
	fixed8();
	fixed8(int, int);

	//class function
	uint16_t bin();
	int m();
	int n();
	void fq(double, int, int);
	double fiq();
	void print_vars();
	void add(fixed8, fixed8);
	void mult(fixed8, fixed8);
	void truncate(fixed8, int, int);
	unsigned int printb();
	void set(uint16_t, int, int);
	void reset();
	void reset(int, int);
	void maxQ(fixed8, fixed8, fixed8, fixed8);

	//operator overload
	fixed8 operator+(const fixed8&);
	fixed8 operator*(const fixed8&);

	//friend function, defined inline
	friend uint16_t get_bin(fixed8);
	friend int get_m(fixed8);
	friend int get_n(fixed8);
	friend fixed8 fq(double, int, int);
	friend double fiq(fixed8);
	friend void print_vars(fixed8);
	friend fixed8 truncate(fixed8, int, int);
	friend uint32_t printb(fixed8);
	friend fixed8 maxQ(fixed8, fixed8, fixed8, fixed8);
	friend fixed8 add_bias(fixed8, fixed8);
};

inline fixed8::fixed8()
{
	_bin = 0;
	_m = 8;
	_n = 0;
}

inline fixed8::fixed8(int m, int n)
{
	_bin = 0;
	_m = m;
	_n = n;
}

inline uint16_t fixed8::bin()
{
	return _bin;
}

inline int fixed8::m()
{
	return _m;
}

inline int fixed8::n()
{
	return _n;
}

inline void fixed8::set(uint16_t x, int m, int n)
{
	_bin = x;
	_m = m;
	_n = n;
}

inline void fixed8::reset()
{
	_bin = 0;
	_m = 8;
	_n = 0;
}

inline void fixed8::reset(int m, int n)
{
	_bin = 0;
	_m = m;
	_n = n;
}

inline void fixed8::maxQ(fixed8 a, fixed8 b, fixed8 c, fixed8 d)
{
	uint16_t temp_max1, temp_max2;
	if (a._bin > b._bin)
		temp_max1 = a._bin;
	else
		temp_max1 = b._bin;
		
	if (c._bin > d._bin)
		temp_max2 = c._bin;
	else
		temp_max2 = d._bin;

	if (temp_max1 > temp_max2)
		_bin = temp_max1;
	else
		_bin = temp_max2;

	_m = a._m;
	_n = a._n;
}

inline uint16_t get_bin(fixed8 x)
{
	return x._bin;
}

inline int get_m(fixed8 x)
{
	return x._m;
}

inline int get_n(fixed8 x)
{
	return x._n;
}

inline fixed8 maxQ(fixed8 a, fixed8 b, fixed8 c, fixed8 d)
{
	fixed8 r;
	uint16_t temp_max1, temp_max2;
	if (a._bin > b._bin)
		temp_max1 = a._bin;
	else
		temp_max1 = b._bin;

	if (c._bin > d._bin)
		temp_max2 = c._bin;
	else
		temp_max2 = d._bin;

	if (temp_max1 > temp_max2)
		r._bin = temp_max1;
	else
		r._bin = temp_max2;

	r._m = a._m;
	r._n = a._n;
	return r;
}

inline fixed8 fq(double x, int m, int n)
{
	fixed8 r;
	r._bin = qNum8::fast_quantizer(x, m, n);
	r._m = m;
	r._n = n;
	return r;
}

inline double fiq(fixed8 x)
{
	return qNum8::fast_inverse_quantizer(x._bin, x._m, x._n);
}
inline void print_vars(fixed8 x)
{
	printf("%d\n",x._m);
	printf("%d\n",x._n);
}


inline fixed8 truncate(fixed8 x, int rm, int rn)
{
	uint16_t ref = (1 << (x._m + x._n - 1));
	fixed8 r;
	r._m = rm;
	r._n = rn;
	if ((x._bin  & ref) == 0)
		r._bin = qNum8::truncation(x._bin, x._m, x._n, rm, rn);
	else
		r._bin = 0;
	
	// r._bin = qNum8::truncation(x._bin, x._m, x._n, rm, rn);
	// r._m = rm;
	// r._n = rn;
	return r;
}

inline uint32_t printb(fixed8 x)
{
	return qNum8::print_bin(x._bin, (x._m + x._n));
}

inline fixed8 fixed8::operator+(const fixed8& y)
{
	fixed8 r;
	// if (y._bin == 0)
	// 	r._bin = this->_bin;
	// else
	r._bin = qNum8::sum(this->_bin, y._bin, this->_m, this->_n, y._m, y._n);

	r._m = this->_m;
	r._n = this->_n;
	return r;
}

inline fixed8 fixed8::operator*(const fixed8& y)
{
	fixed8 r;
	if ((this->_m + this->_n == 4) && (y._m + y._n == 4))
	{
		r._bin = qNum8::mult_8b(this->_bin, y._bin, this->_m, this->_n, y._m, y._n);
		r._m = this->_m + y._m + 2;
		r._n = this->_n + y._n;
	}
	else
	{
		r._bin = qNum8::mult_general(this->_bin, y._bin, this->_m, this->_n, y._m, y._n);
		r._m = this->_m + y._m;
		r._n = this->_n + y._n;
	}
	return r;
}

inline fixed8 add_bias(fixed8 x, fixed8 y)
{
	fixed8 r;
	r._bin = qNum8::sum_b(x._bin, y._bin, x._m, x._n, y._m, y._n);
	r._m = x._m;
	r._n = x._n;
	return r;
}
