#include "elasticdl/cc/common/type.h"

namespace elasticdl {
namespace common {
size_t GetElementSize(ElemType i) {
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

}  // namespace common
}  // namespace elasticdl