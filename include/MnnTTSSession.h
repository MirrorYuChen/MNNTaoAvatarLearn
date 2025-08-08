/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 18:53:22
 * @Contact: 2458006466@qq.com
 * @Description: MnnTTSSession
 */
#pragma once

#include "Api.h"

#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <tuple>

#include "MnnTTSSessionConfig.h"

NAMESPACE_BEGIN
typedef std::vector<int16_t> Audio;
class API MnnTTSSessionImplBase {
public:
  virtual ~MnnTTSSessionImplBase() = default;
  virtual std::tuple<int, Audio> Process(const std::string &text) = 0;

private:
  int sample_rate_ = 16000;
};

class API MnnTTSSession {
public:
  MnnTTSSession(const std::string &cfg_dir);
  std::tuple<int, Audio> Process(const std::string &text);
  void WriteAudioToFile(const Audio &audio_data, const std::string &output_file_path);

private:
  int sample_rate_;
  std::shared_ptr<MnnTTSSessionImplBase> impl_;
};

NAMESPACE_END
