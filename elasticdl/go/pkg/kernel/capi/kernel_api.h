#ifndef ELASTICDL_PKG_KERNEL_CAPI_KERNEL_API_H_
#define ELASTICDL_PKG_KERNEL_CAPI_KERNEL_API_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void SGD(float* grad, float* param, float lr, long long size);

void Momentum(float* grad,
              float* param,
              float* velocity,
              float mu,
              bool nesterov,
              float lr,
              long long size);

void Adam(float* grad,
          float* param,
          float* m,
          float* v,
          float lr,
          long long size,
          long long step,
          float beta1,
          float beta2,
          float epsilon,
          float* max_square);

void Adagrad(float* grad,
             float* param,
             float* m,
             float lr,
             long long size,
             float epsilon);

#ifdef __cplusplus
}
#endif
#endif
