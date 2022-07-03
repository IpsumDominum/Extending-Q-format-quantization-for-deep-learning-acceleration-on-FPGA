#include "fixed8.h"
#include "qLib8.h"
#include <iostream>

void fixed8::fq(double x, int m, int n)
{
	_bin = qNum8::fast_quantizer(x, m, n);
	_m = m;
	_n = n;
}

double fixed8::fiq()
{
	return qNum8::fast_inverse_quantizer(_bin, _m, _n);
}

void fixed8::add(fixed8 x, fixed8 y)
{
	if (y.bin() == 0)
		_bin = x._bin;
	else
		_bin = qNum8::sum(x._bin, y._bin, x._m, x._n, y._m, y._n);
	_m = x._m;
	_n = x._n;
}

void fixed8::mult(fixed8 x, fixed8 y)
{
	_bin = qNum8::mult_general(x._bin, y._bin, x._m, x._n, y._m, y._n);
	_m = x._m + y._m + 3;
	_n = x._n + y._n;
}

void fixed8::truncate(fixed8 x, int rm, int rn)
{
	_bin = qNum8::truncation(x._bin, x._m, x._n, rm, rn);
	_m = rm;
	_n = rn;
}
void fixed8::print_vars()
{
	printf("%d\n",_m);
	printf("%d\n",_n);
}


uint32_t fixed8::printb()
{
	return qNum8::print_bin(_bin, (_m + _n));
}
