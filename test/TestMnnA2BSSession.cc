/*
 * @Author: chenjingyu
 * @Date: 2025-08-11 10:10:17
 * @Contact: 2458006466@qq.com
 * @Description: MnnA2BSSession
 */
#include "MnnA2BSSession.h"
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

using namespace NAMESPACE;

// 读取音频文件并返回int16_t类型的PCM缓冲区
std::vector<int16_t> LoadAudioToInt16PCM(const std::string &filePath) {
  ma_result result;
  ma_decoder decoder;
  std::vector<int16_t> pcmData;

  // 初始化解码器
  result = ma_decoder_init_file(filePath.c_str(), nullptr, &decoder);
  if (result != MA_SUCCESS) {
    std::cerr << "Failed to initialize decoder. Error: " << result << std::endl;
    return pcmData;
  }

  // 获取音频格式信息
  const ma_format format = decoder.outputFormat;
  const ma_uint32 channels = decoder.outputChannels;
  const ma_uint32 sampleRate = decoder.outputSampleRate;

  std::cout << "Audio format: " << format << ", Channels: " << channels
            << ", Sample rate: " << sampleRate << std::endl;

  // 临时缓冲区（用于读取原始数据，miniaudio默认可能返回float或int32等格式）
  std::vector<float> tempBuffer(4096); // 4096样本的临时缓冲
  ma_uint64 framesRead;

  do {
    // 读取音频数据（以float格式读取，范围[-1.0, 1.0]）
    result = ma_decoder_read_pcm_frames(&decoder, tempBuffer.data(),
                                        tempBuffer.size(), &framesRead);
    if (result != MA_SUCCESS && result != MA_AT_END) {
      std::cerr << "Failed to read audio data. Error: " << result << std::endl;
      break;
    }

    // 将float格式转换为int16_t（范围[-32768, 32767]）
    for (size_t i = 0; i < framesRead * channels; ++i) {
      // 钳位防止溢出
      float scaled = tempBuffer[i] * 32767.0f;
      if (scaled > 32767.0f)
        scaled = 32767.0f;
      if (scaled < -32768.0f)
        scaled = -32768.0f;
      pcmData.push_back(static_cast<int16_t>(scaled));
    }
  } while (framesRead > 0);

  // 清理资源
  ma_decoder_uninit(&decoder);
  return pcmData;
}

int main(int argc, char *argv[]) {
  std::unique_ptr<MnnA2BSSession> session = std::make_unique<MnnA2BSSession>();
  if (session->LoadA2bsResources("../data/a2bs/", "")) {
    std::cout << "Load a2bs recources successed." << std::endl;
  } else {
    std::cout << "Failed to Load a2bs recources." << std::endl;
    return -1;
  }
  auto audio_data = LoadAudioToInt16PCM("./result.wav");
  auto result = session->Process(0, audio_data.data(), audio_data.size(), 44100);
  session->SaveResultDataToBinaryFile(result, 0);

  return 0;
}
