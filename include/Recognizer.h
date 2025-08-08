/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 09:50:56
 * @Contact: 2458006366@qq.com
 * @Description: Recognizer
 */
#pragma once

#include "Api.h"

#include <memory>
#include <string>
#include <vector>

#include "RecognizerConfig.h"

NAMESPACE_BEGIN
class RecognizerImpl;

class API Recognizer {
public:
  explicit Recognizer(const RecognizerConfig &config);

  ~Recognizer();

  bool Process(const std::vector<std::string> &wav_filenames, std::string &result);

private:
  std::unique_ptr<RecognizerImpl> impl_;
};

NAMESPACE_END
