#ifndef ELASTICDL_CC_COMMON_TYPE_H_
#define ELASTICDL_CC_COMMON_TYPE_H_

#include "glog/logging.h"

namespace elasticdl {
namespace common {

enum class ElemType {
  Float32,
  Float64,
  Int32,
};

size_t GetElementSize(ElemType i);

template <typename T>
bool IsType(ElemType e) {
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

}  // namespace common
}  // namespace elasticdl
#endif