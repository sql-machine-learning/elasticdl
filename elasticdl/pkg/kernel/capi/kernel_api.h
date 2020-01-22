#ifndef ELASTICDL_PKG_KERNEL_CAPI_KERNEL_API_H_
#define ELASTICDL_PKG_KERNEL_CAPI_KERNEL_API_H_

#ifdef __cplusplus
extern "C" {
#endif

void SGD(float* grad, float* param, double lr, long long size);

void Adam(float* grad,
          float* param,
          float* m,
          float* v,
          double lr,
          long long size,
          long long step,
          double beta1,
          double beta2,
          double epsilon,
          float* max_square);

#ifdef __cplusplus
}
#endif
#endif
