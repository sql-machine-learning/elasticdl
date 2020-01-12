#include "elasticdl/cc/common/tensor.h"

#include "gtest/gtest.h"

using namespace elasticdl::common;

TEST(Tensor, Init) {
  Tensor t1("t1", ElemType::Int32, {10, 10});
  EXPECT_EQ("t1", t1.name());
  auto* data = t1.mutable_data<int>();
  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < 10; j++) {
      data[i * 10 + j] = i * 10 + j;
    }
  }
  EXPECT_EQ(t1.at<int>(10), 10);
}

TEST(Tensor, RowSparseTensor) {
  std::vector<int64_t> indices = {0, 1, 3, 5, 7};
  std::vector<int64_t> dim = {5, 2};
  std::vector<float> data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  Tensor t("t", ElemType::Float32, dim, data.data(), indices);

  EXPECT_EQ("t", t.name());
  EXPECT_EQ(t.GetHeight(), 5);
  EXPECT_EQ(t.GetWidth(), 2);
  EXPECT_EQ(t.indices()[3], 5);
  EXPECT_FLOAT_EQ(t.at<float>(2), 0.3);
}