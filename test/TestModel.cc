/*
 * @Author: chenjingyu
 * @Date: 2025-08-07 17:45:45
 * @Contact: 2458006366@qq.com
 * @Description: TestModel
 */
#include "Core/Model.h"

using namespace NAMESPACE;

int main(int argc, char *argv[]) {
  ModelConfig cfg;
  cfg.debug = true;
  cfg.encoder = "../data/encoder.mnn";
  cfg.decoder = "../data/decoder.mnn";
  cfg.joiner = "../data/joiner.mnn";
  cfg.tokens = "../data/tokens.txt";

  Model model(cfg);



  return 0;
}
