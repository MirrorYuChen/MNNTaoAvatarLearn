   /*
 * @Author: chenjingyu
 * @Date: 2025-08-07 14:49:44
 * @Contact: 2458006366@qq.com
 * @Description: RecognizerConfig
 */
#pragma once

#include "Api.h"
#include "ParseOptions.h"

#include <vector>
#include <string>

NAMESPACE_BEGIN
struct API RecognizerResult {
  /// Recognition results.
  /// For English, it consists of space separated words.
  /// For Chinese, it consists of Chinese words without spaces.
  /// Example 1: "hello world"
  /// Example 2: "你好世界"
  std::string text;

  /// Decoded results at the token level.
  /// For instance, for BPE-based models it consists of a list of BPE tokens.
  std::vector<std::string> tokens;

  /// timestamps.size() == tokens.size()
  /// timestamps[i] records the time in seconds when tokens[i] is decoded.
  std::vector<float> timestamps;
  std::vector<float> ys_probs;  //< log-prob scores from ASR model

  std::vector<int32_t> words;

  /// ID of this segment
  /// When an endpoint is detected, it is incremented
  int32_t segment = 0;

  /// Starting time of this segment.
  /// When an endpoint is detected, it will change
  float start_time = 0;

  /// True if the end of this segment is reached
  bool is_final = false;

  /** Return a json string.
   *
   * The returned string contains:
   *   {
   *     "text": "The recognition result",
   *     "tokens": [x, x, x],
   *     "timestamps": [x, x, x],
   *     "ys_probs": [x, x, x],
   *     "segment": x,
   *     "start_time": x,
   *     "is_final": true|false
   *   }
   */
  [[nodiscard]] std::string AsJsonString() const;
};

struct API FeatureExtractorConfig {
// Sampling rate used by the feature extractor. If it is different from
  // the sampling rate of the input waveform, we will do resampling inside.
  int32_t sampling_rate = 16000;

  // num_mel_bins
  //
  // Note: for mfcc, this value is also for num_mel_bins.
  // The actual feature dimension is actuall num_ceps
  int32_t feature_dim = 80;

  // minimal frequency for Mel-filterbank, in Hz
  float low_freq = 20.0f;

  // maximal frequency of Mel-filterbank
  // in Hz; negative value is subtracted from Nyquist freq.:
  // i.e. for sampling_rate 16000 / 2 - 400 = 7600Hz
  //
  // Please see
  // https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/fbank.py#L27
  // and
  // https://github.com/k2-fsa/sherpa-mnn/issues/514
  float high_freq = -400.0f;

  // dithering constant, useful for signals with hard-zeroes in non-speech parts
  // this prevents large negative values in log-mel filterbanks
  //
  // In k2, audio samples are in range [-1..+1], in kaldi the range was
  // [-32k..+32k], so the value 0.00003 is equivalent to kaldi default 1.0
  //
  float dither = 0.0f; // dithering disabled by default

  // Set internally by some models, e.g., paraformer sets it to false.
  // This parameter is not exposed to users from the commandline
  // If true, the feature extractor expects inputs to be normalized to
  // the range [-1, 1].
  // If false, we will multiply the inputs by 32768
  bool normalize_samples = true;

  bool snip_edges = false;
  float frame_shift_ms = 10.0f; // in milliseconds.
  float frame_length_ms = 25.0f; // in milliseconds.
  bool is_librosa = false;
  bool remove_dc_offset = true; // Subtract mean of wave before FFT.
  float preemph_coeff = 0.97f; // Preemphasis coefficient.
  std::string window_type = "povey"; // e.g. Hamming window

  // For models from NeMo
  // This option is not exposed and is set internally when loading models.
  // Possible values:
  // - per_feature
  // - all_features (not implemented yet)
  // - fixed_mean (not implemented)
  // - fixed_std (not implemented)
  // - or just leave it to empty
  // See
  // https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py#L59
  // for details
  std::string nemo_normalize_type;

  // for MFCC
  int32_t num_ceps = 13;
  bool use_energy = true;

  std::string ToString () const;

  void Register (ParseOptions *po);
};

struct API ModelConfig {
  ModelConfig() = default;
  ModelConfig(std::string encoder, std::string decoder, std::string joiner) :
    encoder(std::move(encoder)), decoder(std::move(decoder)), joiner(std::move(joiner)) {}

  void Register(ParseOptions *po);

  [[nodiscard]] bool Validate() const;

  [[nodiscard]] std::string ToString() const;

  std::string encoder;
  std::string decoder;
  std::string joiner;
  std::string tokens;
  int32_t num_threads = 1;
  int32_t warm_up = 0;
  bool debug = false;

  /// if tokens_buf is non-empty,
  /// the tokens will be loaded from the buffer instead of from the
  /// "tokens" file
  std::string tokens_buf;
};

struct API EndpointRule {
  // If True, for this endpointing rule to apply there must
  // be nonsilence in the best-path traceback.
  // For decoding, a non-blank token is considered as non-silence
  bool must_contain_nonsilence = true;
  // This endpointing rule requires duration of trailing silence
  // (in seconds) to be >= this value.
  float min_trailing_silence = 2.0;
  // This endpointing rule requires utterance-length (in seconds)
  // to be >= this value.
  float min_utterance_length = 0.0f;

  EndpointRule () = default;

  EndpointRule (bool must_contain_nonsilence, float min_trailing_silence,
                float min_utterance_length)
    : must_contain_nonsilence(must_contain_nonsilence),
      min_trailing_silence(min_trailing_silence),
      min_utterance_length(min_utterance_length) {
  }

  [[nodiscard]] std::string ToString () const;
};

struct API EndpointConfig {
  // For default setting,
  // rule1 times out after 2.4 seconds of silence, even if we decoded nothing.
  // rule2 times out after 1.2 seconds of silence after decoding something.
  // rule3 times out after the utterance is 20 seconds long, regardless of
  // anything else.
  EndpointRule rule1;
  EndpointRule rule2;
  EndpointRule rule3;

  void Register (ParseOptions *po);

  EndpointConfig ()
    : rule1{false, 2.4, 0}, rule2{true, 1.2, 0}, rule3{false, 0, 20} {
  }

  EndpointConfig (const EndpointRule &rule1, const EndpointRule &rule2,
                  const EndpointRule &rule3)
    : rule1(rule1), rule2(rule2), rule3(rule3) {
  }

  [[nodiscard]] std::string ToString () const;
};

struct API RecognizerConfig {
  RecognizerConfig() = default;
  explicit RecognizerConfig(
    FeatureExtractorConfig feature_extractor_config,
    ModelConfig model_config,
    EndpointConfig endpoint_config,
    bool enable_endpoint,
    float blank_penalty, float temperature_scale,
    const std::string &rule_fsts, const std::string &rule_fars) :
    feature_extractor_config(std::move(feature_extractor_config)),
    model_config(std::move(model_config)), endpoint_config(std::move(endpoint_config)),
    enable_endpoint(enable_endpoint), blank_penalty(blank_penalty), temperature_scale(temperature_scale) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  [[nodiscard]] std::string ToString() const;

  FeatureExtractorConfig feature_extractor_config;
  ModelConfig model_config;
  EndpointConfig endpoint_config;
  bool enable_endpoint = true;

  float blank_penalty = 0.0;

  float temperature_scale = 2.0;
};

NAMESPACE_END
