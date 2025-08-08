/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 09:51:09
 * @Contact: 2458006366@qq.com
 * @Description: Recognizer
 */
#include "Core/Recognizer.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "Base/WaveReader.h"
#include "Core/RecognizerImpl.h"
#include "Base/Logger.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "Utils/FileUtils.h"
#include "Utils/TextUtils.h"

NAMESPACE_BEGIN
typedef struct {
  std::unique_ptr<Stream> stream;
  float duration;
  float elapsed_seconds;
} StreamWrapper;

Recognizer::Recognizer(const RecognizerConfig &config)
    : impl_(RecognizerImpl::Create(config)) {}

Recognizer::~Recognizer() = default;

bool Recognizer::Process(const std::vector<std::string> &wav_filenames, std::string &result) {
  std::vector<StreamWrapper> ss;
  const auto begin = std::chrono::steady_clock::now();
  std::vector<float> durations;

  // 1.process input wav files
  for (const auto &wav_filename : wav_filenames) {
    int32_t sampling_rate = -1;

    bool is_ok = false;
    const std::vector<float> samples = ReadWave(wav_filename, &sampling_rate, &is_ok);

    if (!is_ok) {
      LogError("Failed to read: {}.", wav_filename);
      return false;
    }

    const float duration = samples.size() / static_cast<float>(sampling_rate);

    auto s = impl_->CreateStream();
    s->AcceptWaveform(sampling_rate, samples.data(), samples.size());

    std::vector<float> tail_paddings(static_cast<int>(0.8 * sampling_rate));
    // Note: We can call AcceptWaveform() multiple times.
    s->AcceptWaveform(sampling_rate, tail_paddings.data(),
                      tail_paddings.size());

    // Call InputFinished() to indicate that no audio samples are available
    s->InputFinished();
    ss.push_back({std::move(s), duration, 0});
  }

  // 2.process streams
  std::vector<Stream *> ready_streams;
  for (;;) {
    ready_streams.clear();
    for (auto &s : ss) {
      const auto p_ss = s.stream.get();
      if (impl_->IsReady(p_ss)) {
        ready_streams.push_back(p_ss);
      } else if (s.elapsed_seconds == 0) {
        const auto end = std::chrono::steady_clock::now();
        const float elapsed_seconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
                .count() /
            1000.;
        s.elapsed_seconds = elapsed_seconds;
      }
    }

    if (ready_streams.empty()) {
      break;
    }

    impl_->DecodeStreams(ready_streams.data(), ready_streams.size());
  }

  // 3.get result
  std::ostringstream os;
  for (const auto &s : ss) {
    const float rtf = s.elapsed_seconds / s.duration;
    os << std::setprecision(2) << "Elapsed seconds: " << s.elapsed_seconds
       << ", Audio duration (s): " << s.duration
       << ", Real time factor (RTF) = " << s.elapsed_seconds << "/"
       << s.duration << " = " << rtf << "\n";
    const auto r = impl_->GetResult(s.stream.get());
    os << r.text << "\n";
    result = r.AsJsonString();
    os << result << "\n\n";
  }
  LogInfo("processed result: {}", os.str());
  return true;
}

NAMESPACE_END
