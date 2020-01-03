#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "glog/logging.h"

namespace elasticdl {
namespace common {

enum class ElemType {
  FP32,
  FP64,
  INT32,
};

static size_t GetElementSize(ElemType i) {
  switch (i) {
    case ElemType::FP32:
      return sizeof(float);
    case ElemType::FP64:
      return sizeof(double);
    case ElemType::INT32:
      return sizeof(int32_t);
  }
  LOG(FATAL) << "Invalid Type";
}

template <class T>
static bool IsType(ElemType e) {
  switch (e) {
    case ElemType::FP32:
      return std::is_same<T, float>::value;
    case ElemType::FP64:
      return std::is_same<T, double>::value;
    case ElemType::INT32:
      return std::is_same<T, int32_t>::value;
  }
  LOG(FATAL) << "Invalid Type";
}

class Tensor {
 public:
  Tensor(const std::vector<size_t>& dim, ElemType type);

  ~Tensor();

  template <class T>
  T* GetRawDataPointer() {
    assert(IsType<T>(element_type_));
    return reinterpret_cast<T*>(data_);
  }

  size_t GetSize();

  template <class T>
  T& at(size_t index) {
    auto* data = GetRawDataPointer<T>();
    return data[index];
  }

 private:
  std::string name_;
  char* data_{nullptr};
  ElemType element_type_;
  // TODO(qijun) use small vector instead
  std::vector<size_t> dim_;
};

}  // namespace common
}  // namespace elasticdl