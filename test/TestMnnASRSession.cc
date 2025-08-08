/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 13:08:18
 * @Contact: 2458006366@qq.com
 * @Description: TestMnnASRSession
 */
#include "MnnASRSession.h"
#include "ParseOptions.h"
#include "Base/WaveReader.h"

#include <iomanip>
#include <iostream>

using namespace NAMESPACE;

int main(int argc, char *argv[]) {
const char *kUsageMessage = R"usage(
Usage:

(1) Streaming transducer

  ./TestMnnASRSession \
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
  MnnASRSessionConfig config;

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

  MnnASRSession MnnASRSession(config);
  std::vector<std::string> wav_filenames;
  for (int32_t i = 1; i <= po.NumArgs(); ++i) {
    wav_filenames.emplace_back(po.GetArg(i));
  }
  std::string result;
  if (MnnASRSession.Process(wav_filenames, result)) {
    std::cout << "prcessed succeed." << std::endl;
    std::cout << result << std::endl;
  } else {
    std::cout << "processed failed." << std::endl;
  }

  return 0;
}