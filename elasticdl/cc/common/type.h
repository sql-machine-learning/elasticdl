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

}  // namespace common
}  // namespace elasticdl
#endif