/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 18:29:19
 * @Contact: 2458006466@qq.com
 * @Description: MnnLLMSession
 */
#pragma once

#include "Api.h"

#include <chrono>
#include <string>
#include <vector>
#include <functional>

#include <MNN/llm/llm.hpp>
#include <json.hpp>

NAMESPACE_BEGIN
using MNN::Transformer::Llm;
using MNN::Transformer::ChatMessage;
using MNN::Transformer::ChatMessages;
using nlohmann::json;

class API MnnLLMSession {
public:
  using Callback = std::function<bool(const std::string &, bool)>;
  MnnLLMSession(std::string model_cfg_path, json config, json extra_config,
             std::vector<std::string> string_history=std::vector<std::string>());
  void Reset();
  void Load();
  ~MnnLLMSession();
  std::string getDebugInfo();
  const MNN::Transformer::LlmContext *Response(
    const std::string &prompt,
    const Callback &cb = nullptr
  );
  void setMaxNewTokens(int i);

  void setSystemPrompt(std::string system_prompt);

  void setAssistantPrompt(const std::string &assistant_prompt);

  std::string getSystemPrompt() const;

  void clearHistory(int numToKeep = 1);

private:
  std::string response_string_for_debug_{};
  std::string model_cfg_path_;
  ChatMessages history_{};
  json extra_cfg_{};
  json cfg_{};
  bool is_r1_{false};
  bool stop_requested_{false};
  bool generate_text_end_{false};
  bool keep_history_{true};
  std::vector<float> waveform{};
  Llm *llm_{nullptr};
  std::string prompt_string_for_debug{};
  int max_new_tokens_{2048};
  std::string system_prompt_;
  json current_cfg_{};
};


NAMESPACE_END
