/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 18:58:24
 * @Contact: 2458006466@qq.com
 * @Description: MnnTTSSession
 */
#include "MnnTTSSession.h"

#include <codecvt> // For std::wstring_convert and std::codecvt_utf8
#include <locale>
#include <mutex>

#include "Base/WaveFile.h"

#include "Bertvits2/MnnBertvits2TTSSessionImpl.h"

NAMESPACE_BEGIN
MnnTTSSession::MnnTTSSession(const std::string &cfg_dir) {
  std::string config_json_path = cfg_dir + "/config.json";
  auto config = MnnTTSSessionConfig(config_json_path);
  auto model_type = config.model_type_;
  auto model_path = cfg_dir + "/" + config.model_path_;
  auto assset_folder = cfg_dir + "/" + config.asset_folder_;
  auto cache_folder = cfg_dir + "/" + config.cache_folder_;
  sample_rate_ = config.sample_rate_;

  if (model_type == "bertvits") {
    impl_ = std::make_shared<MnnBertvits2TTSSessionImpl>(
        assset_folder, model_path, cache_folder);
  } else {
    throw std::runtime_error("Invalid model type");
    return;
  }
}
std::tuple<int, Audio> MnnTTSSession::Process(const std::string &text) {
  return impl_->Process(text);
}

void MnnTTSSession::WriteAudioToFile(const Audio &audio_data,
                                     const std::string &output_file_path) {
  std::ofstream audioFile(output_file_path, std::ios::binary);

  // Write WAV
  WriteWavHeader(sample_rate_, 2, 1, (int32_t)audio_data.size(), audioFile);

  audioFile.write((const char *)audio_data.data(),
                  sizeof(int16_t) * audio_data.size());
}

NAMESPACE_END
