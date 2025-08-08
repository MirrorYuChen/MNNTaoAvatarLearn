   /*
 * @Author: chenjingyu
 * @Date: 2024-09-20 11:06:48
 * @Contact: 2458006466@qq.com
 * @Description: Logger
 */
#include "Base/Logger.h"

#include <cstdlib>

#if SPDLOG_VER_MAJOR >= 1
#if defined(__ANDROID__)
#include <sinks/android_sink.h>
#else
#include <sinks/stdout_color_sinks.h>
#if defined(_MSC_VER)
#include <sinks/stdout_sinks.h>
#endif
#endif
#endif

#if SPDLOG_VER_MAJOR >= 1 && SPDLOG_VER_MINOR >= 6
#define MIRROR_SPDLOG_HAS_LOAD_ENV_LEVELS 1
#include <cfg/env.h>
#endif

NAMESPACE_BEGIN
static void LoadEnvLevels() {
  auto p = std::getenv("SPDLOG_LEVEL");
  if (p) {
    const std::string str(p);
    if (str == "trace") {
      spdlog::set_level(spdlog::level::trace);
    } else if (str == "debug") {
      spdlog::set_level(spdlog::level::debug);
    } else if (str == "info") {
      spdlog::set_level(spdlog::level::info);
    } else if (str == "warn") {
      spdlog::set_level(spdlog::level::warn);
    } else if (str == "err") {
      spdlog::set_level(spdlog::level::err);
    } else if (str == "critical") {
      spdlog::set_level(spdlog::level::critical);
    } else if (str == "off") {
      spdlog::set_level(spdlog::level::off);
    }
  }
}

std::shared_ptr<spdlog::logger> CreateDefaultLogger() {
#if MIRROR_SPDLOG_HAS_LOAD_ENV_LEVELS
  spdlog::cfg::load_env_levels();
#else
  LoadEnvLevels();
#endif
  constexpr const auto logger_name = "mirror";
#if defined(__ANDROID__)
  return spdlog::android_logger_mt(logger_name);
#elif defined(_MSC_VER)
  return spdlog::stdout_logger_mt(logger_name);
#else
  return spdlog::stdout_color_mt(logger_name);
#endif
}

std::shared_ptr<spdlog::logger> &gLogger() {
  // ! leaky singleton
  static auto ptr = new std::shared_ptr<spdlog::logger>{CreateDefaultLogger()};
  return *ptr;
}

spdlog::logger *getLogger() { return gLogger().get(); }

void setLogger(spdlog::logger *logger) {
  gLogger() = std::shared_ptr<spdlog::logger>(logger, [](auto) {});
}

LogStreamImpl::LogStreamImpl(
  int level, 
  const char *file, 
  int line,
  bool abort_flag
) : level_(level), file_(file), line_(line), abort_flag_(abort_flag) {
  log_stream_ << "[" << file_ << ": " << line_ << "] ";
}

LogStreamImpl::~LogStreamImpl() {
  getLogger()->log(spdlog::level::level_enum(level_),
                   log_stream_.str().c_str());
  if (abort_flag_) {
    abort();
  }
}
std::ostream &LogStreamImpl::stream() { return log_stream_; }

static std::string StringPrintf(const char *format, va_list args) {
  std::string output;
  // 1.try a small buffer and hope it fits
  char space[128];
  va_list args_backup;
  va_copy(args_backup, args);
  int bytes = vsnprintf(space, sizeof(space), format, args_backup);
  va_end(args_backup);

  if ((bytes >= 0) && (static_cast<size_t>(bytes) < sizeof(space))) {
    output.append(space, bytes);
    return output;
  }

  // 2.Repeatedly increase buffer size until it fits.
  int length = sizeof(space);
  while (true) {
    if (bytes < 0) {
      length *= 2;
    } else {
      length = bytes + 1;
    }
    char *buf = new char[length];
    // 2.1.Restore the va_list before we use it again
    va_copy(args_backup, args);
    bytes = vsnprintf(buf, length, format, args_backup);
    va_end(args_backup);

    if ((bytes >= 0) && (bytes < length)) {
      output.append(buf, bytes);
      delete[] buf;
      break;
    } else {
      delete[] buf;
    }
  }
  return output;
}

void LogPrintfImpl(int level, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  std::string msg = StringPrintf(fmt, args);
  va_end(args);

  getLogger()->log(spdlog::level::level_enum(level), msg.c_str());
}

NAMESPACE_END
