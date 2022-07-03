#pragma once
#include <cstdint>

namespace qNum8
{
	uint16_t sum(uint16_t, uint16_t, int, int, int, int);
	uint16_t sum_b(uint16_t, uint16_t, int, int, int, int);
	uint16_t mult_8b(uint16_t x, uint16_t y, int xm, int xn, int ym, int yn);
	uint16_t mult_general(uint16_t, uint16_t, int, int, int, int);
	uint16_t truncation(uint16_t, int, int, int, int);
	uint64_t print_bin(uint16_t, int);
	uint64_t ipow(uint32_t, uint32_t);
	uint16_t fast_quantizer(double, int, int);
	double fast_inverse_quantizer(uint16_t, int, int);
}

