#ifndef ELASTICDL_PKG_KERNEL_CAPI_VECTOR_H_
#define ELASTICDL_PKG_KERNEL_CAPI_VECTOR_H_

#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Vector{
    void* Data;
    size_t Dtype;
    size_t Length;
    size_t ByteLength;
} Vector;

double IndexAtVector(Vector* vec, size_t idx);

void* AddressAt(Vector* vec, size_t idx);

void SetVector(Vector* vec, size_t idx, double val);

#ifdef __cplusplus
}
#endif
#endif
