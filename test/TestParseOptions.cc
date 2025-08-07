/*
 * @Author: chenjingyu
 * @Date: 2025-08-07 10:44:38
 * @Contact: 2458006466@qq.com
 * @Description: TestParseOptions
 */
#include "Base/ParseOptions.h"
#include "Base/RecognizerConfig.h"
#include <sstream>

using namespace NAMESPACE;

int main(int argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Usage:

(1) Streaming transducer
  ./TestParseOptions \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.mnn \
    --decoder=/path/to/decoder.mnn \
    --joiner=/path/to/joiner.mnn \
    --num-threads=2 \
    --decoding-method=greedy_search \
    /path/to/foo.wav [bar.wav foobar.wav ...]
)usage";

  ParseOptions po(kUsageMessage);
  RecognizerConfig cfg;
  cfg.Register(&po);
  po.Read(argc, argv);

  if (po.NumArgs() < 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "%s\n", cfg.ToString().c_str());

  if (!cfg.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  for (int32_t i = 1; i <= po.NumArgs(); i++) {
    const std::string wav_filename = po.GetArg(i);
    LogInfo("wave filename: {}", wav_filename);
  }

  return 0;
}


