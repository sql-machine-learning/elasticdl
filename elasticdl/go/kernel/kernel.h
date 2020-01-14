#ifndef ELASTICDL_KERNEL_H_
#define ELASTICDL_KERNEL_H_

#include <stdint.h>

void SGD(float* grad, float* param, double lr, int64_t size);
#endif