#ifndef ELASTICDL_PKG_KERNEL_CAPI_KERNEL_API_H_
#define ELASTICDL_PKG_KERNEL_CAPI_KERNEL_API_H_

#ifdef __cplusplus
extern "C" {
#endif

void SGD(float* grad, float* param, float lr, long long size);

#ifdef __cplusplus
}
#endif
#endif
