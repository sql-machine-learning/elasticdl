#include "kernel_api.h"

#include <cmath>
#include <eigen3/Eigen/Dense>

void SGD(float* grad, float* param, float lr, long long size) {
  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> eg{
      grad, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ep{
      param, static_cast<Eigen::Index>(size)};

  ep -= lr * eg;
}

void Momentum(float* grad,
              float* param,
              float* velocity,
              float mu,
              bool nesterov,
              float lr,
              long long size) {
  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> eg{
      grad, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ep{
      param, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ev{
      velocity, static_cast<Eigen::Index>(size)};

  ev = mu * ev + eg;
  if (nesterov) {
    ep -= lr * (eg + mu * ev);
  } else {
    ep -= lr * ev;
  }
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

  lr *= sqrt(1 - pow(beta2, step)) / (1 - pow(beta1, step));

  if (max_square != NULL) {
    Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ems{
        max_square, static_cast<Eigen::Index>(size)};
    ems = ems.cwiseMax(ev);
    ep -= lr * em / (ems.sqrt() + epsilon);
  } else {
    ep -= lr * em / (ev.sqrt() + epsilon);
  }
}

void Adagrad(float* grad,
             float* param,
             float* m,
             float lr,
             long long size,
             float epsilon) {
  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> eg{
      grad, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> ep{
      param, static_cast<Eigen::Index>(size)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> em{
      m, static_cast<Eigen::Index>(size)};

  em += eg.square();
  ep -= lr * eg / (em.sqrt() + epsilon);
}