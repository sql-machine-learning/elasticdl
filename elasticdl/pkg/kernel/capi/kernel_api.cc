#include "kernel_api.h"

#include <cmath>
#include <eigen3/Eigen/Dense>

void AddToInt8(void* a, void* b, long long size) {
  Eigen::Map<Eigen::Array<char, 1, Eigen::Dynamic>> ea{
      (char*)a, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<char, 1, Eigen::Dynamic>> eb{
      (char*)b, static_cast<Eigen::Index>(size)};

  ea += eb;
}

void AddToInt16(void* a, void* b, long long size) {
  Eigen::Map<Eigen::Array<short, 1, Eigen::Dynamic>> ea{
      (short*)a, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<short, 1, Eigen::Dynamic>> eb{
      (short*)b, static_cast<Eigen::Index>(size)};

  ea += eb;
}

void AddToInt32(void* a, void* b, long long size) {
  Eigen::Map<Eigen::Array<int, 1, Eigen::Dynamic>> ea{
      (int*)a, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<int, 1, Eigen::Dynamic>> eb{
      (int*)b, static_cast<Eigen::Index>(size)};

  ea += eb;
}

void AddToInt64(void* a, void* b, long long size) {
  Eigen::Map<Eigen::Array<long long, 1, Eigen::Dynamic>> ea{
      (long long*)a, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<long long, 1, Eigen::Dynamic>> eb{
      (long long*)b, static_cast<Eigen::Index>(size)};

  ea += eb;
}

void AddToFloat32(void* a, void* b, long long size) {
  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ea{
      (float*)a, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> eb{
      (float*)b, static_cast<Eigen::Index>(size)};

  ea += eb;
}

void AddToFloat64(void* a, void* b, long long size) {
  Eigen::Map<Eigen::Array<double, 1, Eigen::Dynamic>> ea{
      (double*)a, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<double, 1, Eigen::Dynamic>> eb{
      (double*)b, static_cast<Eigen::Index>(size)};

  ea += eb;
}

void (*AddToFunc[])(void*, void*, long long) = {NULL,
                                                AddToInt8,
                                                AddToInt16,
                                                AddToInt32,
                                                AddToInt64,
                                                NULL,
                                                AddToFloat32,
                                                AddToFloat64,
                                                NULL};

void AddTo(void* a, void* b, long long size, int dtypeFlag) {
  AddToFunc[dtypeFlag](a, b, size);
}

void SGDFloat32(void* grad, void* param, float lr, long long size) {
  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> eg{
      (float*)grad, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ep{
      (float*)param, static_cast<Eigen::Index>(size)};

  ep -= lr * eg;
}

void SGDFloat64(void* grad, void* param, float lr, long long size) {
  Eigen::Map<Eigen::Array<double, 1, Eigen::Dynamic>> eg{
      (double*)grad, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<double, 1, Eigen::Dynamic>> ep{
      (double*)param, static_cast<Eigen::Index>(size)};

  ep -= lr * eg;
}

void (*SGDFunc[])(void*, void*, float, long long) = {
    NULL, NULL, NULL, NULL, NULL, NULL, SGDFloat32, SGDFloat64, NULL};

void SGD(void* grad, void* param, float lr, long long size, int dtype) {
  SGDFunc[dtype](grad, param, lr, size);
}

void AdamFloat32(void* grad,
                 void* param,
                 void* m,
                 void* v,
                 float lr,
                 long long size,
                 long long step,
                 float beta1,
                 float beta2,
                 float epsilon,
                 void* max_square) {
  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> eg{
      (float*)grad, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ep{
      (float*)param, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> em{
      (float*)m, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ev{
      (float*)v, static_cast<Eigen::Index>(size)};

  em = beta1 * em + (1.0 - beta1) * eg;

  ev = beta2 * ev + (1.0 - beta2) * eg.square();

  if (max_square != NULL) {
    Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ems{
        (float*)max_square, static_cast<Eigen::Index>(size)};
    ems = ems.cwiseMax(ev);
    ep -= lr * em / (1.0 - pow(beta1, step)) /
          ((ems / (1.0 - pow(beta2, step))).sqrt() + epsilon);
  } else {
    ep -= lr * em / (1.0 - pow(beta1, step)) /
          ((ev / (1.0 - pow(beta2, step))).sqrt() + epsilon);
  }
}

void AdamFloat64(void* grad,
                 void* param,
                 void* m,
                 void* v,
                 float lr,
                 long long size,
                 long long step,
                 float beta1,
                 float beta2,
                 float epsilon,
                 void* max_square) {
  Eigen::Map<Eigen::Array<double, 1, Eigen::Dynamic>> eg{
      (double*)grad, static_cast<Eigen::Index>(size)};
  Eigen::Map<Eigen::Array<double, 1, Eigen::Dynamic>> ep{
      (double*)param, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<double, 1, Eigen::Dynamic>> em{
      (double*)m, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<double, 1, Eigen::Dynamic>> ev{
      (double*)v, static_cast<Eigen::Index>(size)};

  em = beta1 * em + (1.0 - beta1) * eg;

  ev = beta2 * ev + (1.0 - beta2) * eg.square();

  if (max_square != NULL) {
    Eigen::Map<Eigen::Array<double, 1, Eigen::Dynamic>> ems{
        (double*)max_square, static_cast<Eigen::Index>(size)};
    ems = ems.cwiseMax(ev);
    ep -= lr * em / (1.0 - pow(beta1, step)) /
          ((ems / (1.0 - pow(beta2, step))).sqrt() + epsilon);
  } else {
    ep -= lr * em / (1.0 - pow(beta1, step)) /
          ((ev / (1.0 - pow(beta2, step))).sqrt() + epsilon);
  }
}

void (*AdamFunc[])(void*,
                   void*,
                   void*,
                   void*,
                   float,
                   long long,
                   long long,
                   float,
                   float,
                   float,
                   void*) = {
    NULL, NULL, NULL, NULL, NULL, NULL, AdamFloat32, AdamFloat64, NULL};

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
          void* max_square,
          int dtype) {
  AdamFunc[dtype](
      grad, param, m, v, lr, size, step, beta1, beta2, epsilon, max_square);
}