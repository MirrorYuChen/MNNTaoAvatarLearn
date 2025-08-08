/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 18:51:11
 * @Contact: 2458006466@qq.com
 * @Description: MnnTTSConfig
 */
#pragma once

#include "Api.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <json.hpp>

namespace fs = std::filesystem;

NAMESPACE_BEGIN
class API MnnTTSSessionConfig {
public:
  explicit MnnTTSSessionConfig(const std::string &config_file_path);

  template <typename T>
  T getValueFromJson(const nlohmann::json &j, const std::string &key) const {
    if (!j.contains(key)) {
      throw std::runtime_error("Missing key in config.json: '" + key + "'");
    }
    try {
      return j.at(key).get<T>();
    } catch (const nlohmann::json::exception &e) {
      throw std::runtime_error("Type mismatch for key '" + key +
                               "': " + e.what());
    }
  }

private:
  nlohmann::json raw_config_data_;

public:
  std::string model_type_;
  std::string model_path_;
  std::string asset_folder_;
  std::string cache_folder_;
  int sample_rate_;
};

NAMESPACE_END
