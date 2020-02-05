#include "kernel_api.h"

#include <cmath>
#include <eigen3/Eigen/Dense>

void AddTo(float* a, float* b, long long size) {
  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ea{
      a, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> eb{
      b, static_cast<Eigen::Index>(size)};

  a += b
}

void SGD(float* grad, float* param, float lr, long long size) {
  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> eg{
      grad, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ep{
      param, static_cast<Eigen::Index>(size)};

  ep -= lr * eg;
}

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
          float* max_square) {
  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> eg{
      grad, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ep{
      param, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> em{
      m, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ev{
      v, static_cast<Eigen::Index>(size)};

  em = beta1 * em + (1.0 - beta1) * eg;

  ev = beta2 * ev + (1.0 - beta2) * eg.square();

  if (max_square != NULL) {
    Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ems{
        max_square, static_cast<Eigen::Index>(size)};
    ems = ems.cwiseMax(ev);
    ep -= lr * em / (1.0 - pow(beta1, step)) /
          ((ems / (1.0 - pow(beta2, step))).sqrt() + epsilon);
  } else {
    ep -= lr * em / (1.0 - pow(beta1, step)) /
          ((ev / (1.0 - pow(beta2, step))).sqrt() + epsilon);
  }
}