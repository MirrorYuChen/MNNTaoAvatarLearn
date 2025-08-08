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

#include "Base/Endpoint.h"
#include "Base/Features.h"
#include "Base/ParseOptions.h"
#include "Base/RecognizerConfig.h"
#include "Core/Model.h"
#include "Core/Stream.h"

NAMESPACE_BEGIN
class RecognizerImpl;

class API Recognizer {
public:
  explicit Recognizer(const RecognizerConfig& config);

  ~Recognizer();

  /// Create a stream for decoding.
  std::unique_ptr<Stream> CreateStream() const;

  /**
   * Return true if the given stream has enough frames for decoding.
   * Return false otherwise
   */
  bool IsReady(Stream *s) const;

  /** Decode a single stream. */
  void DecodeStream(Stream *s) const {
    Stream *ss[1] = {s};
    DecodeStreams(ss, 1);
  }

  /** Decode multiple streams in parallel
   *
   * @param ss Pointer array containing streams to be decoded.
   * @param n Number of streams in `ss`.
   */
  void DecodeStreams(Stream **ss, int32_t n) const;

  RecognizerResult GetResult(Stream *s) const;

  // Return true if we detect an endpoint for this stream.
  // Note: If this function returns true, you usually want to
  // invoke Reset(s).
  bool IsEndpoint(Stream *s) const;

  // Clear the state of this stream. If IsEndpoint(s) returns true,
  // after calling this function, IsEndpoint(s) will return false
  void Reset(Stream *s) const;

private:
  std::unique_ptr<RecognizerImpl> impl_;
};

NAMESPACE_END
