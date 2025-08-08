  /*
 * @Author: chenjingyu
 * @Date: 2024-05-27 14:37:45
 * @Contact: 2458006466@qq.com
 * @Description: Logger
 */
#pragma once

#include "Api.h"
#include <spdlog.h>
#include <chrono>
#include <memory>
#include <sstream>
#include <iostream>
#include <stdarg.h>
#include <string>

#if OS_WINDOWS
#define FILENAME                                                               \
  (strrchr(__FILE__, '\\') ? (strrchr(__FILE__, '\\') + 1) : __FILE__)
#else
#define FILENAME                                                               \
  (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1) : __FILE__)
#endif

// Honor spdlog settings if supported
#if defined(SPDLOG_ACTIVE_LEVEL) && defined(SPDLOG_LEVEL_OFF)

#define MIRROR_LEVEL_TRACE SPDLOG_LEVEL_TRACE
#define MIRROR_LEVEL_DEBUG SPDLOG_LEVEL_DEBUG
#define MIRROR_LEVEL_INFO SPDLOG_LEVEL_INFO
#define MIRROR_LEVEL_WARN SPDLOG_LEVEL_WARN
#define MIRROR_LEVEL_ERROR SPDLOG_LEVEL_ERROR
#define MIRROR_LEVEL_CRITICAL SPDLOG_LEVEL_CRITICAL
#define MIRROR_LEVEL_OFF SPDLOG_LEVEL_OFF

#if !defined(MIRROR_ACTIVE_LEVEL)
#define MIRROR_ACTIVE_LEVEL SPDLOG_ACTIVE_LEVEL
#endif

#else

#define MIRROR_LEVEL_TRACE 0
#define MIRROR_LEVEL_DEBUG 1
#define MIRROR_LEVEL_INFO 2
#define MIRROR_LEVEL_WARN 3
#define MIRROR_LEVEL_ERROR 4
#define MIRROR_LEVEL_CRITICAL 5
#define MIRROR_LEVEL_OFF 6

#if !defined(MIRROR_ACTIVE_LEVEL)
#define MIRROR_ACTIVE_LEVEL MIRROR_LEVEL_INFO
#endif

#endif

#ifdef SPDLOG_LOGGER_CALL
#define MIRROR_LOG(level, ...)                                                 \
  SPDLOG_LOGGER_CALL(NAMESPACE::getLogger(), level, __VA_ARGS__)
#else
#define MIRROR_LOG(level, ...) NAMESPACE::getLogger()->log(level, __VA_ARGS__)
#endif

#if MIRROR_ACTIVE_LEVEL <= MIRROR_LEVEL_TRACE
#define LogTrace(...) MIRROR_LOG(spdlog::level::trace, __VA_ARGS__)
#else
#define LogTrace(...) (void)0;
#endif

#if MIRROR_ACTIVE_LEVEL <= MIRROR_LEVEL_DEBUG
#define LogDebug(...) MIRROR_LOG(spdlog::level::debug, __VA_ARGS__)
#else
#define LogDebug(...) (void)0;
#endif

#if MIRROR_ACTIVE_LEVEL <= MIRROR_LEVEL_INFO
#define LogInfo(...) MIRROR_LOG(spdlog::level::info, __VA_ARGS__)
#else
#define LogInfo(...) (void)0;
#endif

#if MIRROR_ACTIVE_LEVEL <= MIRROR_LEVEL_WARN
#define LogWarn(...) MIRROR_LOG(spdlog::level::warn, __VA_ARGS__)
#else
#define LogWarn(...) (void)0;
#endif

#if MIRROR_ACTIVE_LEVEL <= MIRROR_LEVEL_ERROR
#define LogError(...) MIRROR_LOG(spdlog::level::err, __VA_ARGS__)
#else
#define LogError(...) (void)0;
#endif

#if MIRROR_ACTIVE_LEVEL <= MIRROR_LEVEL_CRITICAL
#define LogCritical(...) MIRROR_LOG(spdlog::level::critical, __VA_ARGS__)
#else
#define LogCritical(...) (void)0;
#endif

// clang-format on
#define CHECK(x)                                                               \
  if (!(x))                                                                    \
  NAMESPACE::LogStreamImpl(4, FILENAME, __LINE__, true).stream()               \
      << "Check failed: " #x << ": " // NOLINT(*)

#define CHECK_BINARY_IMPL(x, cmp, y) CHECK(x cmp y) << x << #cmp << y << " "
#define CHECK_EQ(x, y) CHECK_BINARY_IMPL(x, ==, y)
#define CHECK_NE(x, y) CHECK_BINARY_IMPL(x, !=, y)
#define CHECK_LT(x, y) CHECK_BINARY_IMPL(x, <, y)
#define CHECK_LE(x, y) CHECK_BINARY_IMPL(x, <=, y)
#define CHECK_GT(x, y) CHECK_BINARY_IMPL(x, >, y)
#define CHECK_GE(x, y) CHECK_BINARY_IMPL(x, >=, y)

#define LTrace 0
#define LDebug 1
#define LInfo 2
#define LWarn 3
#define LError 4
#define LCritical 5

#define LogFormat(LEVEL, fmt, ...)                                             \
  NAMESPACE::getLogger()->log(spdlog::level::level_enum(LEVEL),                \
                              "[{}: {}] " fmt, FILENAME, __LINE__,             \
                              ##__VA_ARGS__)

#define LogStream(LEVEL)                                                       \
  NAMESPACE::LogStreamImpl(LEVEL, FILENAME, __LINE__).stream()

#define LogPrintf(LEVEL, fmt, ...)                                             \
  NAMESPACE::LogPrintfImpl(LEVEL, "[%s: %d] " fmt, FILENAME, __LINE__,         \
                           ##__VA_ARGS__)

NAMESPACE_BEGIN
API spdlog::logger *getLogger();
API void setLogger(spdlog::logger *logger);
class API LogStreamImpl {
public:
  explicit LogStreamImpl(int level, const char *file, int line, bool abort_flag = false);
  ~LogStreamImpl();
  std::ostream &stream();

  NOT_ALLOWED_COPY(LogStreamImpl)

private:
  int level_;
  std::string file_;
  int line_;
  bool abort_flag_;
  std::stringstream log_stream_;
};

API void LogPrintfImpl(int level, const char *fmt, ...);
NAMESPACE_END
