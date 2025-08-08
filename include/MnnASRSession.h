/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 09:50:56
 * @Contact: 2458006366@qq.com
 * @Description: MnnASRSession
 */
#pragma once

#include "Api.h"

#include <memory>
#include <string>
#include <vector>

#include "MnnASRSessionConfig.h"

NAMESPACE_BEGIN
class MnnASRSessionImpl;

class API MnnASRSession {
public:
  explicit MnnASRSession(const MnnASRSessionConfig &config);

  ~MnnASRSession();

  bool Process(const std::vector<std::string> &wav_filenames, std::string &result);

private:
  std::unique_ptr<MnnASRSessionImpl> impl_;
};

NAMESPACE_END
