#include "elasticdl/cpp/common/tensor.h"
#include "gtest/gtest.h"

using namespace elasticdl::common;

TEST(Tensor, init) {
  Tensor t1({10, 10}, ElemType::FP32);

  auto* data = t1.GetRawDataPointer<float>();

  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < 10; j++) {
      data[i * 10 + j] = i * 10 + j;
    }
  }

  EXPECT_EQ(t1.at<float>(10), 10);
}