   /*
 * @Author: chenjingyu
 * @Date: 2025-08-07 15:10:09
 * @Contact: 2458006366@qq.com
 * @Description: Features
 */
#pragma once

#include "Api.h"

#include <memory>
#include <string>
#include <vector>

#include "RecognizerConfig.h"

NAMESPACE_BEGIN
class FeatureExtractor {
public:
  explicit FeatureExtractor (const FeatureExtractorConfig &config = {});

  ~FeatureExtractor ();

  /**
     @param sampling_rate The sampling_rate of the input waveform. If it does
                          not equal to  config.sampling_rate, we will do
                          resampling inside.
     @param waveform Pointer to a 1-D array of size n. It must be normalized to
                     the range [-1, 1].
     @param n Number of entries in waveform
   */
  void AcceptWaveform (int32_t sampling_rate, const float *waveform,
                       int32_t n) const;

  /**
   * InputFinished() tells the class you won't be providing any
   * more waveform.  This will help flush out the last frame or two
   * of features, in the case where snip-edges == false; it also
   * affects the return value of IsLastFrame().
   */
  void InputFinished () const;

  int32_t NumFramesReady () const;

  /** Note: IsLastFrame() will only ever return true if you have called
   * InputFinished() (and this frame is the last frame).
   */
  bool IsLastFrame (int32_t frame) const;

  /** Get n frames starting from the given frame index.
   *
   * @param frame_index  The starting frame index
   * @param n  Number of frames to get.
   * @return Return a 2-D tensor of shape (n, feature_dim).
   *         which is flattened into a 1-D vector (flattened in row major)
   */
  std::vector<float> GetFrames (int32_t frame_index, int32_t n) const;

  /// Return feature dim of this extractor
  int32_t FeatureDim () const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

NAMESPACE_END
