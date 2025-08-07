/*
 * @Author: chenjingyu
 * @Date: 2025-08-07 17:21:53
 * @Contact: 2458006366@qq.com
 * @Description: Ops
 */
#include "Core/Ops.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

NAMESPACE_BEGIN
template <typename T /*= float*/>
std::vector<MNN::Express::VARP> Unbind(MNNAllocator *allocator, MNN::Express::VARP value,
                               int32_t dim) {
  std::vector<int> shape = value->getInfo()->dim;
  assert(dim >= 0);
  assert(dim < static_cast<int32_t>(shape.size()));
  int32_t n = static_cast<int32_t>(shape[dim]);
  if (n == 1) {
    std::vector<MNN::Express::VARP> ans;
    ans.push_back(Clone(allocator, value));
    return ans;
  }

  std::vector<int> ans_shape = shape;
  ans_shape[dim] = 1;  // // Unlike torch, we keep the dim to 1

  // allocator tensors
  std::vector<MNN::Express::VARP> ans;
  ans.reserve(n);
  for (int32_t i = 0; i != n; ++i) {
    MNN::Express::VARP t = MNNUtilsCreateTensor<T>(allocator, ans_shape.data(),
                                               ans_shape.size());
    ans.push_back(std::move(t));
  }

  auto leading_size = static_cast<int32_t>(std::accumulate(
      shape.begin(), shape.begin() + dim, 1, std::multiplies<int>()));

  auto trailing_size = static_cast<int32_t>(std::accumulate(
      shape.begin() + dim + 1, shape.end(), 1, std::multiplies<int>()));

  const T *src = value->readMap<T>();

  for (int32_t i = 0; i != leading_size; ++i) {
    for (int32_t k = 0; k != n; ++k) {
      T *dst = ans[k]->writeMap<T>() + i * trailing_size;
      std::copy(src, src + trailing_size, dst);
      src += trailing_size;
    }
  }

  return ans;
}

template std::vector<MNN::Express::VARP> Unbind<float>(MNNAllocator *allocator,
                                               MNN::Express::VARP value,
                                               int32_t dim);

template std::vector<MNN::Express::VARP> Unbind<int>(MNNAllocator *allocator,
                                                 MNN::Express::VARP value,
                                                 int32_t dim);


static bool Compare(const std::vector<int> &a,
                    const std::vector<int> &b, int32_t skip_dim) {
  if (a.size() != b.size()) return false;

  for (int32_t i = 0; i != static_cast<int32_t>(a.size()); ++i) {
    if (i == skip_dim) continue;

    if (a[i] != b[i]) return false;
  }

  return true;
}

static void PrintShape(const std::vector<int> &a) {
  for (auto i : a) {
    fprintf(stderr, "%d ", static_cast<int32_t>(i));
  }
  fprintf(stderr, "\n");
}

template <typename T /*=float*/>
MNN::Express::VARP Cat(MNNAllocator *allocator,
               const std::vector<MNN::Express::VARP > &values, int32_t dim) {
  if (values.size() == 1u) {
    return Clone(allocator, values[0]);
  }

  std::vector<int> v0_shape =
      values[0]->getInfo()->dim;

  int total_dim = v0_shape[dim];

  for (int32_t i = 1; i != static_cast<int32_t>(values.size()); ++i) {
    auto s = values[i]->getInfo()->dim;
    total_dim += s[dim];

    bool ret = Compare(v0_shape, s, dim);
    if (!ret) {
      fprintf(stderr, "Incorrect shape in Cat !\n");

      fprintf(stderr, "Shape for tensor 0: ");
      PrintShape(v0_shape);

      fprintf(stderr, "Shape for tensor %d: ", i);
      PrintShape(s);

      exit(-1);
    }
  }

  std::vector<int> ans_shape;
  ans_shape.reserve(v0_shape.size());
  ans_shape.insert(ans_shape.end(), v0_shape.data(), v0_shape.data() + dim);
  ans_shape.push_back(total_dim);
  ans_shape.insert(ans_shape.end(), v0_shape.data() + dim + 1,
                   v0_shape.data() + v0_shape.size());

  auto leading_size = static_cast<int32_t>(std::accumulate(
      v0_shape.begin(), v0_shape.begin() + dim, 1, std::multiplies<int>()));

  auto trailing_size = static_cast<int32_t>(
      std::accumulate(v0_shape.begin() + dim + 1, v0_shape.end(), 1,
                      std::multiplies<int>()));

  MNN::Express::VARP ans = MNNUtilsCreateTensor<T>(allocator, ans_shape.data(),
                                               ans_shape.size());
  T *dst = ans->writeMap<T>();

  for (int32_t i = 0; i != leading_size; ++i) {
    for (auto value : values) {
      auto this_dim = value->getInfo()->dim[dim];
      const T *src = value->readMap<T>();
      src += i * this_dim * trailing_size;

      std::copy(src, src + this_dim * trailing_size, dst);
      dst += this_dim * trailing_size;
    }
  }

  return ans;
}

template MNN::Express::VARP Cat<float>(MNNAllocator *allocator,
                               const std::vector<MNN::Express::VARP > &values,
                               int32_t dim);

template MNN::Express::VARP Cat<int>(MNNAllocator *allocator,
                                 const std::vector<MNN::Express::VARP > &values,
                                 int32_t dim);

NAMESPACE_END
