#ifndef ELASTICDL_CC_SGD_H_
#define ELASTICDL_CC_SGD_H_

#include <stdint.h>

namespace elasticdl {
void SGD(const float* grad, float* param, double lr, int64_t size);
}  // namespace elasticdl
#endif