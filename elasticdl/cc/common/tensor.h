#ifndef ELASTICDL_CC_COMMON_TENSOR_H_
#define ELASTICDL_CC_COMMON_TENSOR_H_

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "glog/logging.h"

namespace elasticdl {
namespace common {

enum class ElemType {
  Float32,
  Float64,
  Int32,
};

static size_t GetElementSize(ElemType i) {
  switch (i) {
    case ElemType::Float32:
      return sizeof(float);
    case ElemType::Float64:
      return sizeof(double);
    case ElemType::Int32:
      return sizeof(int32_t);
  }
  LOG(FATAL) << "Invalid Type";
}

template <class T>
static bool IsType(ElemType e) {
  switch (e) {
    case ElemType::Float32:
      return std::is_same<T, float>::value;
    case ElemType::Float64:
      return std::is_same<T, double>::value;
    case ElemType::Int32:
      return std::is_same<T, int32_t>::value;
  }
  LOG(FATAL) << "Invalid Type";
}

class Tensor {
 public:
  Tensor(const std::string& name,
         const std::vector<int64_t>& dim,
         const ElemType& type)
      : name_(name), dim_(dim), element_type_(type) {
    int64_t size = GetSize() * GetElementSize(element_type_);
    data_ = new char[size];
  }
  Tensor(const std::string& name,
         const std::vector<int64_t>& dim,
         const ElemType& type,
         void* data,
         void* indices)
      : name_(name),
        dim_(dim),
        element_type_(type),
        data_(reinterpret_cast<char*>(data)),
        indices_(reinterpret_cast<int64_t*>(indices)) {
    is_unowned_ = true;
  }

  ~Tensor() {
    if (!is_unowned()) {
      delete[] data_;
      delete[] indices_;
    }
  }

  bool is_unowned() { return is_unowned_; }

  template <class T>
  T* GetRawDataPointer() {
    assert(IsType<T>(element_type_));
    return reinterpret_cast<T*>(data_);
  }

  int64_t* indices() { return indices_; }

  std::vector<int64_t>& dim() { return dim_; }

  std::string& name() { return name_; }

  int64_t GetHeight();

  int64_t GetWidth();

  int64_t GetSize();

  template <class T>
  T& at(size_t index) {
    auto* data = GetRawDataPointer<T>();
    return data[index];
  }

 private:
  std::string name_;
  std::vector<int64_t> dim_;
  ElemType element_type_;
  char* data_{nullptr};
  int64_t* indices_{nullptr};
  bool is_unowned_{false};
};

}  // namespace common
}  // namespace elasticdl

#endif