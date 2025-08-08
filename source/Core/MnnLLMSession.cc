/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 18:32:49
 * @Contact: 2458006466@qq.com
 * @Description: MnnLLMSession
 */
#include "MnnLLMSession.h"

#include <MNN/MNNForwardType.h>
#include <MNN/expr/ExecutorScope.hpp>
#include <chrono>
#include <utility>
#include <sstream>

#include "Base/Logger.h"

NAMESPACE_BEGIN
// clang-format off
std::string TrimLeadingWhitespace(const std::string &str) {
  auto it = std::find_if(
    str.begin(), str.end(), [](unsigned char ch) { 
      return !std::isspace(ch); 
    }
  );
  return {it, str.end()};
}

std::string getUserString(const char *user_content, bool for_history,
                          bool is_r1) {
  if (is_r1) {
    return "<|User|>" + std::string(user_content) + "<|Assistant|>" +
           (for_history ? "" : "<think>\n");
  } else {
    return user_content;
  }
}

std::string getSystemPromptString(std::string system_prompt, bool is_r1) {
  if (is_r1) {
    return std::string("<|begin_of_sentence|>") + system_prompt;
  } else {
    return system_prompt;
  }
}

std::string DeleteThinkPart(std::string assistant_content) {
  std::size_t think_start = assistant_content.find("<think>");
  if (think_start == std::string::npos) {
    return assistant_content;
  }
  std::size_t think_end = assistant_content.find("</think>", think_start);
  if (think_end == std::string::npos) {
    return assistant_content;
  }
  think_end += std::string("</think>").length();
  assistant_content.erase(think_start, think_end - think_start);
  return assistant_content;
}

std::string getR1AssistantString(std::string assistant_content) {
  std::size_t pos = assistant_content.find("</think>");
  if (pos != std::string::npos) {
    assistant_content.erase(0, pos + std::string("</think>").length());
  }
  return TrimLeadingWhitespace(assistant_content) + "<|end_of_sentence|>";
}

static int getUtf8CharLength(unsigned char byte) {
  if ((byte & 0x80) == 0) return 1;
  if ((byte & 0xE0) == 0xC0) return 2;
  if ((byte & 0xF0) == 0xE0) return 3;
  if ((byte & 0xF8) == 0xF0) return 4;
  return 0;
}

// clang-format on

class Utf8Processor {
public:
  using Callback = std::function<void(const std::string &)>;
  explicit Utf8Processor(Callback cb) : cb_(std::move(cb)) {}

  void Process(const char *data, size_t len) {
    utf8_buffer_.append(data, len);

    size_t i = 0;
    std::string complete_chars;
    while (i < utf8_buffer_.size()) {
      int char_len = getUtf8CharLength(static_cast<unsigned char>(utf8_buffer_[i]));
      if (char_len == 0 || i + char_len > utf8_buffer_.size()) {
        break;
      }
      complete_chars.append(utf8_buffer_, i, char_len);
      i += char_len;
    }
    utf8_buffer_ = utf8_buffer_.substr(i);
    if (!complete_chars.empty()) {
      if (cb_) {
        cb_(complete_chars);
      }
    }
  }

private:
  std::string utf8_buffer_;
  Callback cb_;
};

class StreamBuffer : public std::streambuf {
public:
  using Callback = std::function<void(const char *data, size_t len)>;
  explicit StreamBuffer(Callback cb) : cb_(std::move(cb)) {}

protected:
  std::streamsize xsputn(const char *s, std::streamsize n) override {
    if (cb_) {
      cb_(s, n);
    }
    return n;
  }

private:
  Callback cb_;
};

void MnnLLMSession::Reset() { history_.resize(1); }

MnnLLMSession::MnnLLMSession(std::string model_cfg_path, json config, json extra_cfg,
                       std::vector<std::string> history)
    : model_cfg_path_(std::move(model_cfg_path)), cfg_(std::move(config)),
      extra_cfg_(std::move(extra_cfg)) {
  max_new_tokens_ = cfg_.contains("max_new_tokens")
                        ? cfg_["max_new_tokens"].get<int>()
                        : 2048;
  keep_history_ = !extra_cfg_.contains("keep_history") ||
                  extra_cfg_["keep_history"].get<bool>();
  is_r1_ = extra_cfg_.contains("is_r1") && extra_cfg_["is_r1"].get<bool>();
  system_prompt_ = cfg_.contains("system_prompt")
                       ? cfg_["system_prompt"].get<std::string>()
                       : "You are a helpful assistant.";
  history_.emplace_back("system", getSystemPromptString(system_prompt_, is_r1_));
  if (!history.empty()) {
    for (int i = 0; i < history.size(); i++) {
      if (is_r1_) {
        if (i % 2 == 0) {
          history_.emplace_back("user", getUserString(history[i].c_str(), true, is_r1_));
        } else {
          history_.emplace_back("assistant", getR1AssistantString(history[i]));
        }
      } else {
        history_.emplace_back(i % 2 == 0 ? "user" : "assistant",
                              i % 2 == 0 ? history[i]
                                         : DeleteThinkPart(history[i]));
      }
    }
  }
}

void MnnLLMSession::Load() {
  std::string root_cache_dir_str = extra_cfg_["mmap_dir"];
  bool use_mmap = !extra_cfg_["mmap_dir"].get<std::string>().empty();
  MNN::BackendConfig backendConfig;
  auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
  MNN::Express::ExecutorScope s(executor);
  llm_ = Llm::createLLM(model_cfg_path_);
  json config = cfg_;
  config["use_mmap"] = use_mmap;
  if (use_mmap) {
    std::string temp_dir = root_cache_dir_str;
    config["tmp_path"] = temp_dir;
  }
  if (is_r1_) {
    config["use_template"] = false;
    config["precision"] = "high";
  }
  current_cfg_ = config;
  auto cfg_str = config.dump();
  LogInfo("extra_cfg: {}", cfg_str);
  llm_->set_config(cfg_str);
  LogDebug("dumped config: {}", llm_->dump_config());
  llm_->load();
}

MnnLLMSession::~MnnLLMSession() {
  LogDebug("LIFECYCLE: MnnLLMSession DESTROYED at {}", this);
  if (llm_) {
    delete llm_;
    llm_ = nullptr;
  }
}

const MNN::Transformer::LlmContext *MnnLLMSession::Response(
  const std::string &prompt, const Callback &cb) {
  if (!llm_) {
    return nullptr;
  }

  // 1.combine chat history
  if (!keep_history_) {
    history_.resize(1);
  }
  int curr_size = 0;
  stop_requested_ = false;
  generate_text_end_ = false;
  std::stringstream response_buffer;
  Utf8Processor processor([&response_buffer, &cb, this](const std::string &utf8_content) {
    bool is_eop = utf8_content.find("<eop>") != std::string::npos;
    if (!is_eop) {
      response_buffer << utf8_content;
    } else {
      std::string response_result =  response_buffer.str();
      LogDebug("submitNative Result {}", response_result);

      response_string_for_debug_ = response_result;
      if (is_r1_) {
        auto& last_message = history_.at(history_.size() - 1);
        std::size_t user_think_pos = last_message.second.find("<think>\n");
        if (user_think_pos != std::string::npos) {
            last_message.second.erase(user_think_pos, std::string("<think>\n").length());
        }
        response_result = getR1AssistantString(response_result);
      }
      response_result = TrimLeadingWhitespace(DeleteThinkPart(response_result));
      history_.emplace_back("assistant", response_result);
    }
    if (cb) {
      stop_requested_ = cb(utf8_content, is_eop);
      generate_text_end_ = is_eop;
    }
  });

  StreamBuffer stream_buffer([&processor](const char *data, size_t len){
    processor.Process(data, len);
  });

  std::ostream os(&stream_buffer);
  history_.emplace_back("user", getUserString(prompt.c_str(), false, is_r1_));
  LogDebug("submitNative history count {}", history_.size());
  for (auto &it : history_) {
    prompt_string_for_debug += it.second;
  }
  LogDebug("submitNative prompt_string_for_debug count: {} max_new_tokens_: {}",
    prompt_string_for_debug, max_new_tokens_
  );
  llm_->response(history_, &os, "<eop>", 1);

  curr_size++;
  while (!stop_requested_ && !generate_text_end_ && curr_size < max_new_tokens_) {
    llm_->generate(1);
    curr_size++;
  }
  auto context = llm_->getContext();
  return context;
}

std::string MnnLLMSession::getDebugInfo() {
  return ("last_prompt:\n" + prompt_string_for_debug + "\nlast_response:\n" +
          response_string_for_debug_);
}

void MnnLLMSession::setMaxNewTokens(int i) { max_new_tokens_ = i; }

void MnnLLMSession::setSystemPrompt(std::string system_prompt) {
  system_prompt_ = std::move(system_prompt);
  if (history_.size() > 1) {
    history_.at(0).second = getSystemPromptString(system_prompt_, is_r1_);
  } else {
    history_.emplace_back("system",
                          getSystemPromptString(system_prompt_, is_r1_));
  }
}

void MnnLLMSession::setAssistantPrompt(const std::string &assistant_prompt) {
  current_cfg_["assistant_prompt_template"] = assistant_prompt;
  if (llm_) {
    llm_->set_config(current_cfg_.dump());
  }
  LogDebug("dumped config: {}", llm_->dump_config());
}

void MnnLLMSession::clearHistory(int numToKeep) {
  if (numToKeep < 0) {
    numToKeep = 0;
  }
  if (history_.size() > static_cast<size_t>(numToKeep)) {
    history_.erase(history_.begin() + numToKeep, history_.end());
  }
  // 清空相关缓存
  prompt_string_for_debug.clear();
}

std::string MnnLLMSession::getSystemPrompt() const { return system_prompt_; }

NAMESPACE_END
