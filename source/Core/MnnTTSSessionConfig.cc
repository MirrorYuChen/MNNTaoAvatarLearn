/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 20:11:59
 * @Contact: 2458006466@qq.com
 * @Description: MnnTTSSessionConfig
 */
#include "MnnTTSSessionConfig.h"

NAMESPACE_BEGIN
MnnTTSSessionConfig::MnnTTSSessionConfig(const std::string &config_json_path) {
  // 检查文件是否存在且是常规文件
  if (!fs::exists(config_json_path) || !fs::is_regular_file(config_json_path)) {
    throw std::runtime_error(
        "Config file not found or is not a regular file: " + config_json_path);
  }

  std::ifstream file(config_json_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open config file: " + config_json_path);
  }

  try {
    file >> raw_config_data_; // 直接从文件流解析JSON
  } catch (const nlohmann::json::parse_error &e) {
    throw std::runtime_error("Error parsing config.json (" + config_json_path +
                             "): " + e.what());
  } catch (const std::exception &e) {
    throw std::runtime_error("Error reading config.json (" + config_json_path +
                             "): " + e.what());
  }

  try {
    model_type_ = getValueFromJson<std::string>(raw_config_data_, "model_type");
    model_path_ = getValueFromJson<std::string>(raw_config_data_, "model_path");
    asset_folder_ = getValueFromJson<std::string>(raw_config_data_, "asset_folder");
    cache_folder_ = getValueFromJson<std::string>(raw_config_data_, "cache_folder");
    sample_rate_ = getValueFromJson<int>(raw_config_data_, "sample_rate");
  } catch (const std::runtime_error &e) {
    // 捕获 getValueFromJson 抛出的异常，并添加更多上下文信息
    throw std::runtime_error("Error in config file " + config_json_path + ": " +
                             e.what());
  }
}

NAMESPACE_END
