/*
 * @Author: chenjingyu
 * @Date: 2025-08-04 15:31:22
 * @Contact: 2458006466@qq.com
 * @Description: TestMnnTTSSdk
 */
#include "MnnTTSSession.h"

using namespace NAMESPACE;

int main(int argc, char *argv[]) {
  MnnTTSSession sdk{"../data/tts"};
  auto tts = sdk.Process("你好，请问有什么可以帮助你的吗？");
  sdk.WriteAudioToFile(std::get<1>(tts), "result.wav");

  return 0;
}

