/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 13:08:18
 * @Contact: 2458006366@qq.com
 * @Description: TestRecognizer
 */
#include "Core/Recognizer.h"
#include "Base/ParseOptions.h"
#include "Base/WaveReader.h"

#include <iomanip>

using namespace NAMESPACE;

typedef struct {
  std::unique_ptr<Stream> stream;
  float duration;
  float elapsed_seconds;
} StreamWrapper;

int main(int argc, char *argv[]) {
const char *kUsageMessage = R"usage(
Usage:

(1) Streaming transducer

  ./TestRecognizer \
    --tokens=../data/tokens.txt \
    --encoder=../data/encoder.mnn \
    --decoder=../data/decoder.mnn \
    --joiner=../data/joiner.mnn \
    --num-threads=2 \
    /path/to/foo.wav [bar.wav foobar.wav ...]

Note: It supports decoding multiple files in batches

Default value for num_threads is 2.
Valid values for decoding_method: greedy_search (default), modified_beam_search.
Valid values for provider: cpu (default), cuda, coreml.
foo.wav should be of single channel, 16-bit PCM encoded wave file; its
sampling rate can be arbitrary and does not need to be 16kHz.

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";

  ParseOptions po(kUsageMessage);
  RecognizerConfig config;

  config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() < 1) {
    po.PrintUsage();
    fprintf(stderr, "Error! Please provide at lease 1 wav file\n");
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  Recognizer recognizer(config);

  std::vector<StreamWrapper> ss;

  const auto begin = std::chrono::steady_clock::now();
  std::vector<float> durations;

  for (int32_t i = 1; i <= po.NumArgs(); ++i) {
    const std::string wav_filename = po.GetArg(i);
    int32_t sampling_rate = -1;

    bool is_ok = false;
    const std::vector<float> samples = ReadWave(wav_filename, &sampling_rate, &is_ok);

    if (!is_ok) {
      fprintf(stderr, "Failed to read '%s'\n", wav_filename.c_str());
      return -1;
    }

    const float duration = samples.size() / static_cast<float>(sampling_rate);

    auto s = recognizer.CreateStream();
    s->AcceptWaveform(sampling_rate, samples.data(), samples.size());

    std::vector<float> tail_paddings(static_cast<int>(0.8 * sampling_rate));
    // Note: We can call AcceptWaveform() multiple times.
    s->AcceptWaveform(sampling_rate, tail_paddings.data(),
                      tail_paddings.size());

    // Call InputFinished() to indicate that no audio samples are available
    s->InputFinished();
    ss.push_back({std::move(s), duration, 0});
  }

  std::vector<Stream *> ready_streams;
  for (;;) {
    ready_streams.clear();
    for (auto &s : ss) {
      const auto p_ss = s.stream.get();
      if (recognizer.IsReady(p_ss)) {
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

    recognizer.DecodeStreams(ready_streams.data(), ready_streams.size());
  }

  std::ostringstream os;
  for (int32_t i = 1; i <= po.NumArgs(); ++i) {
    const auto &s = ss[i - 1];
    const float rtf = s.elapsed_seconds / s.duration;

    os << po.GetArg(i) << "\n";
    os << "Number of threads: " << config.model_config.num_threads << ", "
       << std::setprecision(2) << "Elapsed seconds: " << s.elapsed_seconds
       << ", Audio duration (s): " << s.duration
       << ", Real time factor (RTF) = " << s.elapsed_seconds << "/"
       << s.duration << " = " << rtf << "\n";
    const auto r = recognizer.GetResult(s.stream.get());
    os << r.text << "\n";
    os << r.AsJsonString() << "\n\n";
  }

  std::cerr << os.str();

  return 0;
}