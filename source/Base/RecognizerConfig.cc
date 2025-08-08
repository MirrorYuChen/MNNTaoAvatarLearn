   /*
 * @Author: chenjingyu
 * @Date: 2025-08-07 14:49:59
 * @Contact: 2458006366@qq.com
 * @Description: RecognizerConfig
 */
#include "RecognizerConfig.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

NAMESPACE_BEGIN
namespace {
/// Helper for `OnlineRecognizerResult::AsJsonString()`
template <typename T>
std::string VecToString(const std::vector<T> &vec, int32_t precision = 6) {
  std::ostringstream oss;
  if (precision != 0) {
    oss << std::fixed << std::setprecision(precision);
  }
  oss << "[";
  std::string sep = "";
  for (const auto &item : vec) {
    oss << sep << item;
    sep = ", ";
  }
  oss << "]";
  return oss.str();
}

/// Helper for `OnlineRecognizerResult::AsJsonString()`
template <>  // explicit specialization for T = std::string
std::string VecToString<std::string>(const std::vector<std::string> &vec,
                                     int32_t) {  // ignore 2nd arg
  std::ostringstream oss;
  oss << "[";
  std::string sep = "";
  for (const auto &item : vec) {
    oss << sep << std::quoted(item);
    sep = ", ";
  }
  oss << "]";
  return oss.str();
}
}  // namespace

std::string RecognizerResult::AsJsonString() const {
  std::ostringstream os;
  os << "{ ";
  os << "\"text\": " << std::quoted(text) << ", ";
  os << "\"tokens\": " << VecToString(tokens) << ", ";
  os << "\"timestamps\": " << VecToString(timestamps, 2) << ", ";
  os << "\"segment\": " << segment << ", ";
  os << "\"words\": " << VecToString(words, 0) << ", ";
  os << "\"start_time\": " << std::fixed << std::setprecision(2) << start_time
     << ", ";
  os << "\"is_final\": " << (is_final ? "true" : "false");
  os << "}";
  return os.str();
}

void RecognizerConfig::Register(ParseOptions *po) {
  feature_extractor_config.Register(po);
  model_config.Register(po);
  endpoint_config.Register(po);

  po->Register("enable-endpoint", &enable_endpoint,
               "True to enable endpoint detection. False to disable it.");
  po->Register("blank-penalty", &blank_penalty,
               "The penalty applied on blank symbol during decoding. "
               "Note: It is a positive value. "
               "Increasing value will lead to lower deletion at the cost"
               "of higher insertions. "
               "Currently only applicable for transducer models.");
  po->Register("temperature-scale", &temperature_scale,
               "Temperature scale for confidence computation in decoding.");
}

bool RecognizerConfig::Validate() const {
  return model_config.Validate();
}

std::string RecognizerConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineRecognizerConfig(";
  os << "feature_extractor_config=" << feature_extractor_config.ToString() << ", ";
  os << "model_config=" << model_config.ToString() << ", ";
  os << "endpoint_config=" << endpoint_config.ToString() << ", ";
  os << "enable_endpoint=" << (enable_endpoint ? "True" : "False") << ", ";
  os << "blank_penalty=" << blank_penalty << ", ";
  os << "temperature_scale=" << temperature_scale << "\")";

  return os.str();
}

NAMESPACE_END
