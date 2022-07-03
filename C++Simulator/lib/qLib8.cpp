#include "qLib8.h"
#include <cstdint>
#include <iostream>
#include <cmath>
#include <stdexcept>

using namespace qNum8;

uint16_t qNum8::sum(uint16_t x, uint16_t y, int xm, int xn, int ym, int yn)
{
	if (xm != ym || xn != yn)
	{
		printf("ERROR: Invalid input from function 'sum'.\n");
		printf("CAUSE: The Q(m,n) of the two numbers must be match.");
		printf("xm: %d xn: %d\n", xm, xn);
		printf("ym: %d yn:%d",ym, yn);
		exit(1);
	}
	else
	{
		uint16_t mask = (1 << (xm + xn)) - 1;
		uint16_t temp = (x + y) & mask;
		// uint16_t ref = 0b1000000000000000;
		uint16_t ref = 1 << (xm + xn - 1);
		if (((x & ref) != 0) && ((y & ref) != 0) && ((temp & ref) == 0))
		{				
			//If x is negative, y is negative, and temp is not...
			//printf("underflow\n");
			return ref;
		}
		else if (((x & ref) == 0) && ((y & ref) == 0) && ((temp & ref) != 0))
		{
			//If x is positive, y is positive, and temp is negative...
			//printf("overflow\n");
			return (ref - 1);
		}
		else
			return temp;
	}
}

uint16_t qNum8::sum_b(uint16_t x, uint16_t y, int xm, int xn, int ym, int yn)
{
	if (xm != ym || xn != yn)
	{
		printf("ERROR: Invalid input from function 'sum'.\n");
		printf("CAUSE: The Q(m,n) of the two numbers must be match.");
		exit(1);
	}
	else
	{
		uint16_t bin_max = (1 << (xm + xn)) - 1; //1111
		uint16_t ref = 1 << (xm + xn - 1); //1000

		uint16_t temp = (x + y) & bin_max;
		if (((x & ref) != 0) && ((y & ref) != 0) && ((temp & ref) == 0))
			return ref;
		else if (((x & ref) == 0) && ((y & ref) == 0) && ((temp & ref) != 0))
			return (ref - 1);
		else
			return temp;
	}
}

uint16_t qNum8::mult_8b(uint16_t x, uint16_t y, int xm, int xn, int ym, int yn)
{
	int kx = xm + xn;
	int ky = ym + yn;
	uint16_t refx = (1 << (kx - 1));
	uint16_t refy = (1 << (ky - 1));
	uint16_t all_max = -1;
	uint16_t sign_x = all_max << (kx);
	uint16_t sign_y = all_max << (ky);
	uint16_t r_mask = 0b1111111111;

	if ((x & refx) != 0)
		x = x + sign_x; //sign extension if negative
	if ((y & refy) != 0)
		y = y + sign_y; //sign extension if negative
	return (x * y) & r_mask;
}

uint16_t qNum8::mult_general(uint16_t x, uint16_t y, int xm, int xn, int ym, int yn)
{
	int kx = xm + xn;
	int ky = ym + yn;
	uint16_t refx = (1 << (kx - 1));
	uint16_t refy = (1 << (ky - 1));
	uint16_t all_max = -1;

	uint16_t sign_x = all_max << (kx);  //1111111111110000   (x= Q(2,2))
	uint16_t sign_y = all_max << (ky);
	uint16_t r_mask = ~(all_max << (kx+ky));

	if ((x & refx) != 0)
		x = x + sign_x; //sign extension if negative
		// x =  1111111111111xxxx
	if ((y & refy) != 0)
		y = y + sign_y; //sign extension if negative
	// x =   11111111|11111xxxx
	// y =   00000000|0000yyyyy
	//mask = 00000000|111111111
	return (x * y) & r_mask;
}
// uint16_t qNum8::truncation(uint16_t x, int mi, int ni, int mo, int no)
// {
// 	int ndiff = ni - no;
// 	int mdiff = mi - mo;
// 	uint16_t bin_max = (1 << (mo + no)) - 1; //1111

// 	uint16_t bin_max2 = (1 << (mo + no - 1)) - 1; //0111 Max for the output
// 	uint16_t bin_min = bin_max2 + 1; //1000 //Min for the output
// 	uint16_t upBound, loBound;
// 	uint16_t ref = (1 << (mi + ni - 1)); //10000000
// 	uint16_t i_max = (1 << (mi + ni)) - 1; //11111111

// 	if (ndiff >= 0 && mdiff >= 0)
// 	{	
// 		//i = (4,4)
// 		//o = (2,2)
// 		upBound = (bin_max2 << ndiff); //e.g.00011100 for Q(4.4)
// 		//bin_max = 0111
// 		//011100
// 		loBound = (bin_min << ndiff);  //e.g.0010 | 0000 for Q(4.4)
// 		//Min for output = 1000
// 		//Input is 4,4
// 		//Min for input =  0010 | 0000
// 		//Example input =  0010 | 0000
// 	}
// 	else
// 	{
// 		printf("ERROR: Invalid input from function 'qNum::truncation'.\n");
// 		printf("CAUSE: Not enought bits for truncation.");
// 		exit(EXIT_FAILURE);
// 	}
// 	uint16_t temp = x;

// 	if ((x & ref) != 0) // negative case
// 	{
// 		temp = (temp ^ i_max) + 1; //2's com
// 		if (temp >= loBound)
// 			return bin_min;
// 		else
// 		{
// 			temp = temp + (1 << (ndiff - 1));
// 			temp = temp >> ndiff;
// 			temp = temp & bin_max;
// 			x = (temp ^ bin_max) + 1; //2's com again
// 			return x;
// 		}
// 	}
// 	else			//positive case
// 	{
// 		if (x >= upBound)
// 			return bin_max2;
// 		else
// 		{
// 			x = x + (1 << (ndiff - 1));
// 			x = x >> ndiff;
// 			x = x & bin_max;
// 			return x;
// 		}
// 	}
// }

uint16_t qNum8::truncation(uint16_t x, int mi, int ni, int mo, int no)
{
	int ndiff = ni - no;
	int mdiff = mi - mo;
	uint16_t bin_max = (1 << (mo + no)) - 1; //1111
	uint16_t bin_max2 = (1 << (mo + no - 1)) - 1; //0111
	uint16_t upBound;

	if (ndiff >= 0 && mdiff >= 0)
		upBound = (bin_max2 << ndiff);
	else
	{
		printf("ERROR: Invalid input from function 'qNum::truncation'.\n");
		printf("CAUSE: Not enought bits for truncation.");
		exit(EXIT_FAILURE);
	}

	if (x >= upBound)
		return bin_max2;

	x = x + (1 << (ndiff - 1));
 	x = x >> ndiff;
	x = x & bin_max;

	return x;
}

// uint16_t qNum8::truncation(uint16_t x, int mi, int ni, int mo, int no)
// {
// 	int ndiff = ni - no;
// 	int mdiff = mi - mo;
// 	uint16_t temp = x;
// 	uint16_t bin_max = (1 << (mo + no)) - 1; //1111
// 	uint16_t bin_max2 = (1 << (mo + no - 1)) - 1; //0111
// 	uint16_t i_max = (1 << (mi + ni)) - 1; //11111111
// 	uint16_t ref = (1 << (mi + ni - 1));
// 	uint16_t upBound, loBound;

// 	if (ndiff >= 0 && mdiff >= 0)
// 	{
// 		upBound = (bin_max2 << ndiff);
// 		loBound = (i_max << (mo + ni - 1)) & i_max;
// 	}
// 	else
// 	{
// 		printf("ERROR: Invalid input from function 'qNum::truncation'.\n");
// 		printf("CAUSE: Not enought bits for truncation.");
// 		exit(EXIT_FAILURE);
// 	}

// 	if (x >= upBound && x < ref)
// 		return bin_max2;
// 	else if (x <= loBound && x >= ref)
// 		return (bin_max2 + 1);
// 	else
// 	{
// 		if ((x & ref) != 0)
// 		{
// 			temp = (temp - 1) ^ i_max;
// 			temp = temp + (1 << (ndiff - 1));
// 			temp = temp >> ndiff;
// 			temp = temp & bin_max;
// 			temp = (temp ^ bin_max) + 1;
// 			if (temp > bin_max)
// 				temp = 0;
// 		}
// 		else
// 		{
// 			temp = temp + (1 << (ndiff - 1));
// 			temp = temp >> ndiff;
// 			temp = temp & bin_max;
// 			if (temp > bin_max2)
// 				temp = bin_max2;
// 		}
// 		return temp;
// 	}
// }

uint64_t qNum8::print_bin(uint16_t x, int k)
{
	uint64_t temp = 0, r = 0;
	uint16_t ref = (1 << (k - 1));
	for (int j = 0; j < k; j++)
	{
		temp = uint64_t(!!((x << j) & ref));
		r = r + (temp * qNum8::ipow(10, k - j - 1));
	}
	// std::cout << std::setw(k) << std::setfill('0');
	return r;
}

uint64_t qNum8::ipow(uint32_t base, uint32_t exp)
{
	uint64_t result = 1;
	while (true)
	{
		if (exp & 1)
			result *= base;
		exp >>= 1;
		if (!exp)
			break;
		base *= base;
	}

	return result;
}

uint16_t qNum8::fast_quantizer(double x, int m, int n)
{
	double max = pow(2, m - 1) - pow(2, -n);
	double min = -pow(2, m - 1);
	double h_step = pow(2, -n - 1);

	double a_max = max + h_step;
	double a_min = min - h_step;
	double mid = (a_max + a_min) / 2;

	int k = m + n;
	uint16_t ref = (1 << (k - 1));
	uint16_t bin = 0;
	uint16_t bin_max = (1 << k) - 1;

	if (x > mid)
	{
		bin = bin + (1 << (k - 1));
		a_min = mid;
		mid = (mid + a_max) / 2;

		for (int i = 1; i < k; ++i)
		{
			if (x >= mid)
			{
				bin = bin + (1 << (k - i - 1));
				a_min = mid;
				mid = (mid + a_max) / 2;
			}
			else
			{
				a_max = mid;
				mid = (mid + a_min) / 2;
			}
		}
	}
	else
	{
		a_max = mid;
		mid = (mid + a_min) / 2;

		for (int i = 1; i < k; ++i)
		{
			if (x > mid)
			{
				bin = bin + (1 << (k - i - 1));
				a_min = mid;
				mid = (mid + a_max) / 2;
			}
			else
			{
				a_max = mid;
				mid = (mid + a_min) / 2;
			}
		}
	}

	bin = (bin + ref) & bin_max;
	return bin;
}

double qNum8::fast_inverse_quantizer(uint16_t x, int m, int n)
{
	int k = m + n;
	uint16_t temp;
	uint16_t xmax = (1 << k) - 1;
	uint16_t q_min = (1 << (k - 1));
	double step = pow(2, -n);
	double result = 0;

	if (x == q_min)
		return -pow(2, m - 1);
	else if ((x & q_min) != 0)
	{
		temp = (x - 1) ^ xmax;
		for (int i = 0; i < k - 1; ++i)
		{
			if ((temp & 1) == 1)
				result = result + step;
			temp = temp >> 1;
			step = step * 2;
		}
		return -result;
	}
	else
	{
		temp = x;
		for (int i = 0; i < k - 1; ++i)
		{
			if ((temp & 1) == 1)
				result = result + step;
			temp = temp >> 1;
			step = step * 2;
		}
		return result;
	}
}
