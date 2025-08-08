/*
 * @Author: chenjingyu
 * @Date: 2025-08-07 14:49:44
 * @Contact: 2458006366@qq.com
 * @Description: RecognizerConfig
 */
#pragma once

#include "Api.h"
#include "Base/ParseOptions.h"
#include "Base/ModelConfig.h"
#include "Base/Endpoint.h"
#include "Base/Features.h"

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
  std::vector<float> lm_probs;  //< log-prob scores from language model
  //
  /// log-domain scores from "hot-phrase" contextual boosting
  std::vector<float> context_scores;

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
   *     "lm_probs": [x, x, x],
   *     "context_scores": [x, x, x],
   *     "segment": x,
   *     "start_time": x,
   *     "is_final": true|false
   *   }
   */
  [[nodiscard]] std::string AsJsonString() const;
};

struct API RecognizerConfig {
  RecognizerConfig() = default;
  explicit RecognizerConfig(
    FeatureExtractorConfig feature_extractor_config,
    ModelConfig model_config,
    EndpointConfig endpoint_config,
    bool enable_endpoint, int32_t max_active_paths,
    const std::string &hotwords_file, float hotwords_score,
    float blank_penalty, float temperature_scale,
    const std::string &rule_fsts, const std::string &rule_fars) :
    feature_extractor_config(std::move(feature_extractor_config)),
    model_config(std::move(model_config)), endpoint_config(std::move(endpoint_config)),
    enable_endpoint(enable_endpoint), max_active_paths(max_active_paths),
    hotwords_file(hotwords_file), hotwords_score(hotwords_score),
    blank_penalty(blank_penalty), temperature_scale(temperature_scale) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  [[nodiscard]] std::string ToString() const;

  FeatureExtractorConfig feature_extractor_config;
  ModelConfig model_config;
  EndpointConfig endpoint_config;
  bool enable_endpoint = true;

  // used only for modified_beam_search
  int32_t max_active_paths = 4;

  /// used only for modified_beam_search
  std::string hotwords_file;
  float hotwords_score = 1.5;

  float blank_penalty = 0.0;

  float temperature_scale = 2.0;

  /// used only for modified_beam_search, if hotwords_buf is non-empty,
  /// the hotwords will be loaded from the buffered string instead of from the
  /// "hotwords_file"
  std::string hotwords_buf;
};

NAMESPACE_END
