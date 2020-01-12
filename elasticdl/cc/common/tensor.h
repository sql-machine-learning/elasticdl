#ifndef ELASTICDL_CC_COMMON_TENSOR_H_
#define ELASTICDL_CC_COMMON_TENSOR_H_

#include <cstdint>
#include <string>
#include <vector>

#include "elasticdl/cc/common/type.h"

namespace elasticdl {
namespace common {

class Tensor {
 public:
  Tensor(const std::string& name,
         const ElemType& type,
         const std::vector<int64_t>& dim,
         const std::vector<int64_t>& indices = std::vector<int64_t>())
      : name_(name), element_type_(type), dim_(dim), indices_(indices) {
    int64_t size = GetSize() * GetElementSize(element_type_);
    data_ = new char[size];
  }
  Tensor(const std::string& name,
         const ElemType& type,
         const std::vector<int64_t>& dim,
         void* data,
         const std::vector<int64_t>& indices = std::vector<int64_t>())
      : name_(name),
        element_type_(type),
        dim_(dim),
        data_(reinterpret_cast<char*>(data)),
        indices_(indices) {
    is_unowned_ = true;
  }

  ~Tensor() {
    if (!is_unowned()) {
      delete[] data_;
    }
  }

  bool is_unowned() { return is_unowned_; }

  template <typename T>
  T* mutable_data() {
    CHECK(IsType<T>(element_type_));
    return reinterpret_cast<T*>(data_);
  }

  template <typename T>
  const T* data() const {
    CHECK(IsType<T>(element_type_));
    return reinterpret_cast<T*>(data_);
  }

  const std::vector<int64_t>& indices() const { return indices_; }

  const std::vector<int64_t>& dim() const { return dim_; }

  const std::string& name() { return name_; }

  const int64_t GetHeight() const;

  const int64_t GetWidth() const;

  const int64_t GetSize() const;

  template <class T>
  T& at(size_t index) {
    T* d = mutable_data<T>();
    return d[index];
  }

 private:
  std::string name_;
  ElemType element_type_;
  std::vector<int64_t> dim_;
  char* data_{nullptr};
  std::vector<int64_t> indices_;
  bool is_unowned_{false};
};

}  // namespace common
}  // namespace elasticdl

#endif