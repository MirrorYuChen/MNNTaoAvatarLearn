 /*
 * @Author: chenjingyu
 * @Date: 2025-08-07 17:21:14
 * @Contact: 2458006366@qq.com
 * @Description: Ops
 */
#pragma once

#include "Api.h"
#include "Core/MNNUtils.h"

NAMESPACE_BEGIN
/** It is similar to torch.unbind() but we keep the unbind dim to 1 in
 * the output
 *
 * @param allocator Allocator to allocate space for the returned tensor
 * @param value  The tensor to unbind
 * @param dim  The dim along which to unbind the tensor
 *
 * @return Return a list of tensors
 */
template <typename T = float>
std::vector<MNN::Express::VARP> Unbind(MNNAllocator *allocator,
                                       MNN::Express::VARP value, int32_t dim);

/** Cat a list of tensors along the given dim.
 *
 * @param allocator Allocator to allocate space for the returned tensor
 * @param values  Pointer to a list of tensors. The shape of the tensor must
 *                be the same except on the dim to be concatenated.
 * @param dim  The dim along which to concatenate the input tensors
 *
 * @return Return the concatenated tensor
 */
template <typename T = float>
MNN::Express::VARP Cat(MNNAllocator *allocator,
                       const std::vector<MNN::Express::VARP> &values,
                       int32_t dim);
NAMESPACE_END
