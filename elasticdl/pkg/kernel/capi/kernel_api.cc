#include "kernel_api.h"

#include <eigen3/Eigen/Dense>

void SGD(float* grad, float* param, float lr, long long size) {
  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> eg{
      grad, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ep{
      param, static_cast<Eigen::Index>(size)};

  ep -= lr * eg;
}
