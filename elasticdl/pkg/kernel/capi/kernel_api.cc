extern "C" {
#include "kernel_api.h"
}

#include <Eigen/Dense>

void SGD(float* grad, float* param, double lr, int64_t size) {
  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> eg{
      grad, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ep{
      param, static_cast<Eigen::Index>(size)};

  ep -= lr * eg;
}
