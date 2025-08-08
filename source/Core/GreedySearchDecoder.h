/*
 * @Author: chenjingyu
 * @Date: 2025-08-07 18:09:57
 * @Contact: 2458006366@qq.com
 * @Description: GreedySearchDecoder
 */
#pragma once

#include "Api.h"
#include <vector>

#include "Core/Model.h"
#include "Core/MNNUtils.h"

NAMESPACE_BEGIN
class API GreedySearchDecoder {
public:
  GreedySearchDecoder(
    Model *model,
    int32_t unk_id,
    float blank_penalty,
    float temperature_scale
  ) : model_(model),
  unk_id_(unk_id),
  blank_penalty_(blank_penalty),
  temperature_scale_(temperature_scale) {}


  DecoderResult GetEmptyResult() const;

  void StripLeadingBlanks(DecoderResult *r) const;

  void Decode(MNN::Express::VARP encoder_out, std::vector<DecoderResult> *result);

private:
  Model *model_;
  int32_t unk_id_;
  float blank_penalty_;
  float temperature_scale_;
};

NAMESPACE_END
