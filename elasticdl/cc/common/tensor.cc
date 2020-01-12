#include "elasticdl/cc/common/tensor.h"

namespace elasticdl {
namespace common {

const int64_t Tensor::GetSize() const {
  int64_t size = 1;
  for (auto d : dim_) {
    size *= d;
  }
  return size;
}

const int64_t Tensor::GetHeight() const {
  CHECK(indices_.size()) << "GetHeight is used in Row Sparse Tensor";
  CHECK_EQ(dim_.size(), 2) << "Row Sparse Tensor must has two dimensions";
  return dim_[0];
}

const int64_t Tensor::GetWidth() const {
  CHECK(indices_.size()) << "GetHeight is used in Row Sparse Tensor";
  CHECK_EQ(dim_.size(), 2) << "Row Sparse Tensor must has two dimensions";
  return dim_[1];
}

}  // namespace common
}  // namespace elasticdl
