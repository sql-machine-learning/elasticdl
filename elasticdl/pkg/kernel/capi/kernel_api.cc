#include "kernel_api.h"

#include <cmath>
#include <eigen3/Eigen/Dense>

void SGD(float* grad, float* param, double lr, long long size) {
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
          double lr,
          long long size,
          long long step,
          double beta1,
          double beta2,
          double epsilon,
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