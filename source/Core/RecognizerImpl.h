   /*
 * @Author: chenjingyu
 * @Date: 2025-08-08 11:04:15
 * @Contact: 2458006366@qq.com
 * @Description: RecognizerImpl
 */
#pragma once

#include "Api.h"

#include <algorithm>
#include <ios>
#include <memory>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Recognizer.h"
#include "Base/Endpoint.h"
#include "Core/Stream.h"
#include "Base/SymbolTable.h"
#include "Core/GreedySearchDecoder.h"

NAMESPACE_BEGIN
static RecognizerResult Convert(const DecoderResult &src,
                               const SymbolTable &sym_table,
                               float frame_shift_ms, int32_t subsampling_factor,
                               int32_t segment, int32_t frames_since_start) {
  RecognizerResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.tokens.size());

  std::string text;
  for (auto i : src.tokens) {
    auto sym = sym_table[i];

    text.append(sym);

    if (sym.size() == 1 && (sym[0] < 0x20 || sym[0] > 0x7e)) {
      // for bpe models with byte_fallback
      // (but don't rewrite printable characters 0x20..0x7e,
      //  which collide with standard BPE units)
      std::ostringstream os;
      os << "<0x" << std::hex << std::uppercase
         << (static_cast<int32_t>(sym[0]) & 0xff) << ">";
      sym = os.str();
    }

    r.tokens.push_back(std::move(sym));
  }

  if (sym_table.IsByteBpe()) {
    text = sym_table.DecodeByteBpe(text);
  }

  r.text = std::move(text);

  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;
  for (auto t : src.timestamps) {
    float time = frame_shift_s * t;
    r.timestamps.push_back(time);
  }
  r.ys_probs = std::move(src.ys_probs);
  r.segment = segment;
  r.start_time = frames_since_start * frame_shift_ms / 1000.;

  return r;
}

class RecognizerImpl {
public:
  explicit RecognizerImpl(const RecognizerConfig &cfg) :
    cfg_(cfg), model_(Model::Create(cfg.model_config)),
    endpoint_(cfg.endpoint_config) {
    if (!cfg.model_config.tokens_buf.empty()) {
      sym_ = SymbolTable(cfg.model_config.tokens_buf, false);
    } else {
      sym_ = SymbolTable(cfg.model_config.tokens, true);
    }
    if (sym_.Contains("<unk>")) {
      unk_id_ = sym_["<unk>"];
    }
    // model_->SetFeatureDim(cfg.feature_extractor_config.feature_dim);
    decoder_ = std::make_unique<GreedySearchDecoder>(
    model_.get(), unk_id_, cfg.blank_penalty, cfg.temperature_scale
    );
  }

  static std::unique_ptr<RecognizerImpl> Create(const RecognizerConfig &cfg) {
    return std::make_unique<RecognizerImpl>(cfg);
  }

  std::unique_ptr<Stream> CreateStream() const  {
    auto stream = std::make_unique<Stream>(cfg_.feature_extractor_config);
    InitStream(stream.get());
    return stream;
  }

  bool IsReady(Stream *s) const  {
    return s->GetNumProcessedFrames() + model_->ChunkSize() <
           s->NumFramesReady();
  }

  void DecodeStreams(Stream **ss, int32_t n) const {
    int32_t chunk_size = model_->ChunkSize();
    int32_t chunk_shift = model_->ChunkShift();

    int32_t feature_dim = ss[0]->FeatureDim();

    std::vector<DecoderResult> results(n);
    std::vector<float> features_vec(n * chunk_size * feature_dim);
    std::vector<std::vector<MNN::Express::VARP>> states_vec(n);
    std::vector<int> all_processed_frames(n);

    for (int32_t i = 0; i != n; ++i) {
      const auto num_processed_frames = ss[i]->GetNumProcessedFrames();
      std::vector<float> features =
          ss[i]->GetFrames(num_processed_frames, chunk_size);

      // Question: should num_processed_frames include chunk_shift?
      ss[i]->GetNumProcessedFrames() += chunk_shift;

      std::copy(features.begin(), features.end(),
                features_vec.data() + i * chunk_size * feature_dim);

      results[i] = std::move(ss[i]->GetResult());
      states_vec[i] = std::move(ss[i]->GetStates());
      all_processed_frames[i] = num_processed_frames;
    }

    auto memory_info =
        (MNNAllocator*)(nullptr);

    std::array<int, 3> x_shape{n, chunk_size, feature_dim};

    MNN::Express::VARP x = MNNUtilsCreateTensor(memory_info, features_vec.data(),
                                            features_vec.size(), x_shape.data(),
                                            x_shape.size());

    std::array<int, 1> processed_frames_shape{
      static_cast<int>(all_processed_frames.size())};

    MNN::Express::VARP processed_frames = MNNUtilsCreateTensor(
        memory_info, all_processed_frames.data(), all_processed_frames.size(),
        processed_frames_shape.data(), processed_frames_shape.size());

    auto states = model_->StackStates(states_vec);

    auto pair = model_->RunEncoder(std::move(x), std::move(states),
                                   std::move(processed_frames));

    decoder_->Decode(std::move(pair.first), &results);

    std::vector<std::vector<MNN::Express::VARP>> next_states =
        model_->UnStackStates(pair.second);

    for (int32_t i = 0; i != n; ++i) {
      ss[i]->SetResult(results[i]);
      ss[i]->SetStates(std::move(next_states[i]));
    }
  }

  RecognizerResult GetResult(Stream *s) const  {
    DecoderResult decoder_result = s->GetResult();
    decoder_->StripLeadingBlanks(&decoder_result);

    // TODO(fangjun): Remember to change these constants if needed
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = 4;
    auto r = Convert(decoder_result, sym_, frame_shift_ms, subsampling_factor,
                     s->GetCurrentSegment(), s->GetNumFramesSinceStart());
    r.text = RemoveInvalidUtf8Sequences(r.text);
    return r;
  }

  bool IsEndpoint(Stream *s) const  {
    if (!cfg_.enable_endpoint) {
      return false;
    }

    int32_t num_processed_frames = s->GetNumProcessedFrames();

    // frame shift is 10 milliseconds
    float frame_shift_in_seconds = 0.01;

    // subsampling factor is 4
    int32_t trailing_silence_frames = s->GetResult().num_trailing_blanks * 4;

    return endpoint_.IsEndpoint(num_processed_frames, trailing_silence_frames,
                                frame_shift_in_seconds);
  }

  void Reset(Stream *s) const  {
    int32_t context_size = model_->ContextSize();

    {
      // segment is incremented only when the last
      // result is not empty, contains non-blanks and longer than context_size)
      const auto &r = s->GetResult();
      if (!r.tokens.empty() && r.tokens.back() != 0 &&
          r.tokens.size() > context_size) {
        s->GetCurrentSegment() += 1;
      }
    }

    // reset encoder states
    // s->SetStates(model_->GetEncoderInitStates());

    auto r = decoder_->GetEmptyResult();
    auto last_result = s->GetResult();
    // if last result is not empty, then
    // preserve last tokens as the context for next result
    if (static_cast<int32_t>(last_result.tokens.size()) > context_size) {
      std::vector<int> context(last_result.tokens.end() - context_size,
                                   last_result.tokens.end());

      Hypotheses context_hyp({{context, 0}});
      r.hyps = std::move(context_hyp);
      r.tokens = std::move(context);
    }

    s->SetResult(r);

    // Note: We only update counters. The underlying audio samples
    // are not discarded.
    s->Reset();
  }

private:
  void InitStream(Stream *stream) const {
    auto r = decoder_->GetEmptyResult();
    stream->SetResult(r);
    stream->SetStates(model_->GetEncoderInitStates());
  }

private:
  RecognizerConfig cfg_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<GreedySearchDecoder> decoder_;
  Endpoint endpoint_;
  SymbolTable sym_;
  int32_t unk_id_ = -1;
};

NAMESPACE_END
