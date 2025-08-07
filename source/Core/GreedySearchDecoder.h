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
#include "Core/Hypothesis.h"
#include "Core/MNNUtils.h"

NAMESPACE_BEGIN
struct API DecoderResult {
  /// Number of frames after subsampling we have decoded so far
  int32_t frame_offset = 0;

  /// The decoded token IDs so far
  std::vector<int> tokens;

  /// number of trailing blank frames decoded so far
  int32_t num_trailing_blanks = 0;

  /// timestamps[i] contains the output frame index where tokens[i] is decoded.
  std::vector<int32_t> timestamps;

  std::vector<float> ys_probs;
  std::vector<float> lm_probs;
  std::vector<float> context_scores;

  // Cache decoder_out for endpointing
  MNN::Express::VARP decoder_out;

  // used only in modified beam_search
  Hypotheses hyps;

  DecoderResult()
      : tokens{}, num_trailing_blanks(0), decoder_out{nullptr}, hyps{} {}

  DecoderResult(const DecoderResult &other);

  DecoderResult &operator=(const DecoderResult &other);

  DecoderResult(DecoderResult &&other) noexcept;

  DecoderResult &operator=(DecoderResult &&other) noexcept;
};

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

static MNN::Express::VARP BuildDecoderInput(Model *model,
    const std::vector<DecoderResult> &results) {
  int32_t batch_size = static_cast<int32_t>(results.size());
  int32_t context_size = model->ContextSize();
  std::array<int, 2> shape{batch_size, context_size};
  MNN::Express::VARP decoder_input = MNNUtilsCreateTensor<int>(
      model->Allocator(), shape.data(), shape.size());
  int *p = decoder_input->writeMap<int>();

  for (const auto &r : results) {
    const int *begin = r.tokens.data() + r.tokens.size() - context_size;
    const int *end = r.tokens.data() + r.tokens.size();
    std::copy(begin, end, p);
    p += context_size;
  }
  return decoder_input;
}

static MNN::Express::VARP BuildDecoderInput(
  Model *model, const std::vector<Hypothesis> &hyps) {
  int32_t batch_size = static_cast<int32_t>(hyps.size());
  int32_t context_size = model->ContextSize();
  std::array<int, 2> shape{batch_size, context_size};
  MNN::Express::VARP decoder_input = MNNUtilsCreateTensor<int>(
      model->Allocator(), shape.data(), shape.size());
  int *p = decoder_input->writeMap<int>();

  for (const auto &h : hyps) {
    std::copy(h.ys.end() - context_size, h.ys.end(), p);
    p += context_size;
  }
  return decoder_input;
}

NAMESPACE_END
