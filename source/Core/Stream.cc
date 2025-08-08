/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 09:58:00
 * @Contact: 2458006366@qq.com
 * @Description: Stream
 */
#include "Core/Stream.h"

#include <memory>
#include <utility>
#include <vector>

#include "Base/Features.h"

NAMESPACE_BEGIN
class Stream::Impl {
 public:
  explicit Impl(const FeatureExtractorConfig &config)
      : feat_extractor_(config) {}

  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n) {
    feat_extractor_.AcceptWaveform(sampling_rate, waveform, n);
  }

  void InputFinished() const { feat_extractor_.InputFinished(); }

  int32_t NumFramesReady() const {
    return feat_extractor_.NumFramesReady() - start_frame_index_;
  }

  bool IsLastFrame(int32_t frame) const {
    return feat_extractor_.IsLastFrame(frame);
  }

  std::vector<float> GetFrames(int32_t frame_index, int32_t n) const {
    return feat_extractor_.GetFrames(frame_index + start_frame_index_, n);
  }

  void Reset() {
    // we don't reset the feature extractor
    start_frame_index_ += num_processed_frames_;
    num_processed_frames_ = 0;
  }

  int32_t &GetNumProcessedFrames() { return num_processed_frames_; }

  int32_t GetNumFramesSinceStart() const { return start_frame_index_; }

  int32_t &GetCurrentSegment() { return segment_; }

  void SetResult(const DecoderResult &r) { result_ = r; }

  DecoderResult &GetResult() { return result_; }
  int32_t FeatureDim() const { return feat_extractor_.FeatureDim(); }

  void SetStates(std::vector<MNN::Express::VARP> states) {
    states_ = std::move(states);
  }

  std::vector<MNN::Express::VARP> &GetStates() { return states_; }

 private:
  FeatureExtractor feat_extractor_;
  int32_t num_processed_frames_ = 0;  // before subsampling
  int32_t start_frame_index_ = 0;     // never reset
  int32_t segment_ = 0;
  DecoderResult result_;
  KeywordResult prev_keyword_result_;
  KeywordResult keyword_result_;
  KeywordResult empty_keyword_result_;
  std::vector<MNN::Express::VARP> states_;  // states for transducer or ctc models
};

Stream::Stream(const FeatureExtractorConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

Stream::~Stream() = default;

void Stream::AcceptWaveform(int32_t sampling_rate, const float *waveform,
                                  int32_t n) const {
  impl_->AcceptWaveform(sampling_rate, waveform, n);
}

void Stream::InputFinished() const { impl_->InputFinished(); }

int32_t Stream::NumFramesReady() const { return impl_->NumFramesReady(); }

bool Stream::IsLastFrame(int32_t frame) const {
  return impl_->IsLastFrame(frame);
}

std::vector<float> Stream::GetFrames(int32_t frame_index,
                                           int32_t n) const {
  return impl_->GetFrames(frame_index, n);
}

void Stream::Reset() { impl_->Reset(); }

int32_t Stream::FeatureDim() const { return impl_->FeatureDim(); }

int32_t &Stream::GetNumProcessedFrames() {
  return impl_->GetNumProcessedFrames();
}

int32_t Stream::GetNumFramesSinceStart() const {
  return impl_->GetNumFramesSinceStart();
}

int32_t &Stream::GetCurrentSegment() {
  return impl_->GetCurrentSegment();
}

void Stream::SetResult(const DecoderResult &r) {
  impl_->SetResult(r);
}

DecoderResult &Stream::GetResult() {
  return impl_->GetResult();
}

void Stream::SetStates(std::vector<MNN::Express::VARP> states) {
  impl_->SetStates(std::move(states));
}

std::vector<MNN::Express::VARP> &Stream::GetStates() {
  return impl_->GetStates();
}

NAMESPACE_END
