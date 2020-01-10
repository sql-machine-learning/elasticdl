#include "elasticdl/cc/common/tensor.h"

namespace elasticdl {
namespace common {

int64_t Tensor::GetSize() {
  int64_t size = 1;
  for (auto d : dim_) {
    size *= d;
  }
  return size;
}

int64_t Tensor::GetHeight() {
  CHECK(indices_.size()) << "GetHeight is used in Row Sparse Tensor";
  CHECK_EQ(dim_.size(), 2) << "Row Sparse Tensor must has two dimensions";
  return dim_[0];
}

int64_t Tensor::GetWidth() {
  CHECK(indices_.size()) << "GetHeight is used in Row Sparse Tensor";
  CHECK_EQ(dim_.size(), 2) << "Row Sparse Tensor must has two dimensions";
  return dim_[1];
}

}  // namespace common
}  // namespace elasticdl
