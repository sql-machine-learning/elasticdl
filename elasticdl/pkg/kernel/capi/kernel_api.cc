#include "kernel_api.h"

#include <cmath>
#include <eigen3/Eigen/Dense>

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