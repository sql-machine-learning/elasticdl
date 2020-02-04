#include "vector.h"

double IndexAtInt8(Vector* vec, size_t idx){
	return ((char*)(vec->Data))[idx];
}

double IndexAtInt16(Vector* vec, size_t idx){
	return ((short*)(vec->Data))[idx];
}

double IndexAtInt32(Vector* vec, size_t idx){
	return ((int*)(vec->Data))[idx];
}

double IndexAtInt64(Vector* vec, size_t idx){
	return ((long long*)(vec->Data))[idx];
}

double IndexAtFloat32(Vector* vec, size_t idx){
	return ((float*)(vec->Data))[idx];
}

double IndexAtFloat64(Vector* vec, size_t idx){
	return ((double*)(vec->Data))[idx];
}

double((*indexFunc[]))(Vector*, size_t)  = {NULL, IndexAtInt8, IndexAtInt16, IndexAtInt32, IndexAtInt64, NULL, IndexAtFloat32, IndexAtFloat64, NULL};

double IndexAtVector(Vector* vec, size_t idx){
	return indexFunc[vec->Dtype](vec, idx);
}

void* AddressAtInt8(Vector* vec, size_t idx){
	return ((char*)(vec->Data))+idx;
}

void* AddressAtInt16(Vector* vec, size_t idx){
	return ((short*)(vec->Data))+idx;
}

void* AddressAtInt32(Vector* vec, size_t idx){
	return ((int*)(vec->Data))+idx;
}

void* AddressAtInt64(Vector* vec, size_t idx){
	return ((long long*)(vec->Data))+idx;
}

void* AddressAtFloat32(Vector* vec, size_t idx){
	return ((float*)(vec->Data))+idx;
}

void* AddressAtFloat64(Vector* vec, size_t idx){
	return ((double*)(vec->Data))+idx;
}

void*((*addressFunc[]))(Vector*, size_t)  = {NULL, AddressAtInt8, AddressAtInt16, AddressAtInt32, AddressAtInt64, NULL, AddressAtFloat32, AddressAtFloat64, NULL};

void* AddressAt(Vector* vec, size_t idx){
	return addressFunc[vec->Dtype](vec, idx);
}

void SetInt8(Vector* vec, size_t idx, double val){
	((char*)(vec->Data))[idx] = val;
}

void SetInt16(Vector* vec, size_t idx, double val){
	((short*)(vec->Data))[idx] = val;
}

void SetInt32(Vector* vec, size_t idx, double val){
	((int*)(vec->Data))[idx] = val;
}

void SetInt64(Vector* vec, size_t idx, double val){
	((long long*)(vec->Data))[idx] = val;
}

void SetFloat32(Vector* vec, size_t idx, double val){
	((float*)(vec->Data))[idx] = val;
}

void SetFloat64(Vector* vec, size_t idx, double val){
	((double*)(vec->Data))[idx] = val;
}

void((*SetFunc[]))(Vector*, size_t, double)  = {NULL, SetInt8, SetInt16, SetInt32, SetInt64, NULL, SetFloat32, SetFloat64, NULL};

void SetVector(Vector* vec, size_t idx, double val){
	SetFunc[vec->Dtype](vec, idx, val);
}