#ifndef ELASTICDL_PKG_KERNEL_CAPI_KERNEL_API_H_
#define ELASTICDL_PKG_KERNEL_CAPI_KERNEL_API_H_

#ifdef __cplusplus
extern "C" {
#endif

void SGD(void* grad, void* param, float lr);

void Adam(void* grad,
          void* param,
          void* m,
          void* v,
          float lr,
          long long step,
          float beta1,
          float beta2,
          float epsilon,
          void* max_square);

#ifdef __cplusplus
}
#endif
#endif
