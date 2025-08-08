   /*
 * @Author: chenjingyu
 * @Date: 2025-08-07 14:15:07
 * @Contact: 2458006366@qq.com
 * @Description: ModelConfig
 */
#pragma once

#include "Api.h"
#include "Base/ParseOptions.h"
#include "Utils/FileUtils.h"

NAMESPACE_BEGIN
struct ModelConfig {
  ModelConfig() = default;
  ModelConfig(std::string encoder, std::string decoder, std::string joiner) :
    encoder(std::move(encoder)), decoder(std::move(decoder)), joiner(std::move(joiner)) {}

  void Register(ParseOptions *po) {
    po->Register("encoder", &encoder, "Path to encoder.mnn");
    po->Register("decoder", &decoder, "Path to decoder.mnn");
    po->Register("joiner", &joiner, "Path to joiner.mnn");

    po->Register("tokens", &tokens, "Path to tokens.txt");
    po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");
    po->Register("warm-up", &warm_up,
                 "Number of warm-up to run the onnxruntime"
                 "Valid vales are: zipformer2");
    po->Register("debug", &debug,
                 "true to print model information while loading it.");
  }

  [[nodiscard]] bool Validate() const {
    if (!FileExists(encoder)) {
      LogError("encoder: '{}' does not exists", encoder);
      return false;
    }
    if (!FileExists(decoder)) {
      LogError("decoder: '{}' does not exists", decoder);
      return false;
    }
    if (!FileExists(joiner)) {
      LogError("joiner: '{}' does not exists", joiner);
      return false;
    }
    return true;
  }

  [[nodiscard]] std::string ToString() const {
    std::ostringstream os;
    os << "ModelConfig(";
    os << "encoder=\"" << encoder << "\", ";
    os << "decoder=\"" << decoder << "\", ";
    os << "joiner=\"" << joiner << "\")";
    return os.str();
  }

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

NAMESPACE_END
