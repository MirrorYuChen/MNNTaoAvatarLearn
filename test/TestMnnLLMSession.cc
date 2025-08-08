/*
 * @Author: chenjingyu
 * @Date: 2025-08-05 15:49:42
 * @Contact: 2458006466@qq.com
 * @Description: TestMnnLLMSession
 */
#include "MnnLLMSession.h"
#include "Base/Logger.h"

using namespace NAMESPACE;

bool OnProcess(const std::string &response, bool is_eop) {
  if (!is_eop) {
    LogInfo("response: {}", response);
  }
  return false;
}


int main(int argc, char *argv[]) {
  // 1.model config
  json cfg;
  cfg["max_new_tokens"] = 2048;
  cfg["is_r1"] = false;
  cfg["system_prompt"] = "You are a helpful assistant.";
  if (cfg.contains("is_rl") && cfg["is_r1"].get<bool>()) {
    cfg["use_template"] = false;
    cfg["precision"] = "high";
  }
  cfg["temperature"] = 0.6f;
  cfg["topK"] = 20;
  cfg["topP"] = 0.95f;
  cfg["minP"] = 0.05f;
  cfg["mixed_samplers"] = {"topK", "topP", "minP", "temperature"};
  cfg["sampler_type"] = "mixed";
  cfg["precision"] = "high";
  cfg["penalty"] = 1.2;
  LogInfo("cfg: {}.", cfg.dump());
  
  // 2.model extra config
  json extra_cfg;
  extra_cfg["use_mmap"] = false;
  extra_cfg["mmap_dir"] = "./tmp";
  LogInfo("extra cfg: {}", extra_cfg.dump());

  // 3.initialize llm session
  const std::string model_cfg_path = "../data/llm/config.json";
  std::unique_ptr<MnnLLMSession> session = std::make_unique<MnnLLMSession>(
    model_cfg_path, cfg, extra_cfg
  );
  session->Load();
  // 4.interact
  while (true) {
    std::cout << "\nUser: ";
    std::string user_str;
    std::getline(std::cin, user_str);
    if (user_str == "/exit") {
      break;
    }
    if (user_str == "/reset") {
      session->Reset();
      std::cout << "\nA: reset done." << std::endl;
      continue;
    }
    auto context = session->Response(user_str, OnProcess);
    auto assistant_str = context->generate_str;
    std::cout << "\nAssistant: " << assistant_str << std::endl;
  }

  return 0;
}

