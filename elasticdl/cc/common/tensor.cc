#include "elasticdl/cc/common/tensor.h"

namespace elasticdl {
namespace common {

Tensor::Tensor(const std::vector<size_t>& dim, ElemType type) {
  element_type_ = type;
  dim_ = dim;

  size_t size = GetSize() * GetElementSize(element_type_);
  data_ = new char[size];
}

Tensor::~Tensor() { delete data_; }

size_t Tensor::GetSize() {
  size_t size = 1;
  for (auto d : dim_) {
    size *= d;
  }
  return size;
}
}  // namespace common
}  // namespace elasticdl
