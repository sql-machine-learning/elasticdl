#ifndef ELASTICDL_PKG_KERNEL_CAPI_KERNEL_API_H_
#define ELASTICDL_PKG_KERNEL_CAPI_KERNEL_API_H_

#ifdef __cplusplus
extern "C" {
#endif

void AddTo(void* a, void* b, long long size, int dtypeFlag);

void SGD(void* grad, void* param, float lr, long long size, int dtypeFlag);

void Adam(void* grad,
          void* param,
          void* m,
          void* v,
          float lr,
          long long size,
          long long step,
          float beta1,
          float beta2,
          float epsilon,
          void* maxSquare,
          int dtypeFlag);

#ifdef __cplusplus
}
#endif
#endif
