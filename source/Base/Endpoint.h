   /*
 * @Author: chenjingyu
 * @Date: 2025-08-07 15:04:19
 * @Contact: 2458006366@qq.com
 * @Description: Endpoint
 */
#pragma once

#include "Api.h"
#include <string>
#include <vector>

#include "Base/ParseOptions.h"
#include "Base/RecognizerConfig.h"

NAMESPACE_BEGIN
class ParseOptions;

class Endpoint {
public:
  explicit Endpoint (const EndpointConfig &config) : config_(config) {
  }

  /// This function returns true if this set of endpointing rules thinks we
  /// should terminate decoding.
  [[nodiscard]] bool IsEndpoint (int32_t num_frames_decoded, int32_t trailing_silence_frames,
                                 float frame_shift_in_seconds) const;

private:
  EndpointConfig config_;
};

NAMESPACE_END
