#ifndef ELASTICDL_PKG_KERNEL_CAPI_KERNEL_API_H_
#define ELASTICDL_PKG_KERNEL_CAPI_KERNEL_API_H_

#ifdef __cplusplus
extern "C" {
#endif

void AddTo(float* a, float* b, long long size);

void SGD(float* grad, float* param, float lr, long long size);

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

#ifdef __cplusplus
}
#endif
#endif
