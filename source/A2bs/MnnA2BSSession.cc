/*
 * @Author: chenjingyu
 * @Date: 2025-08-11 09:26:58
 * @Contact: 2458006466@qq.com
 * @Description: MnnA2BSSession
 */
#include "MnnA2BSSession.h"
#include "A2bs/AudioTo3dgsBlendShape.h"
#include "Base/Logger.h"
#include "Utils/A2BSUtils.h"

#include <chrono>

NAMESPACE_BEGIN
using clk = std::chrono::system_clock;

MnnA2BSSession *MnnA2BSSession::instance_ = nullptr;
MnnA2BSSession::MnnA2BSSession() { instance_ = this; }

MnnA2BSSession::~MnnA2BSSession() {
  audio_to_bs_ = nullptr;
  if (instance_ == this) {
    instance_ = nullptr;
  }
}

bool MnnA2BSSession::LoadA2bsResources(const char *res_path,
                                       const char *temp_path) {
  if (!audio_to_bs_) {
    audio_to_bs_ = std::make_shared<AudioTo3DGSBlendShape>(res_path, temp_path, false);
  }
  if (!audio_to_bs_) {
    LogError("Failed to create MnnA2BSSession.");
    return false;
  }
  return true;
}

void MnnA2BSSession::SaveResultDataToBinaryFile(
    const AudioToBlendShapeData &data, int index) {
  std::string filename = "./a2bs_" + std::to_string(index) + ".bin";
  std::ofstream outfile(filename, std::ios::binary);
  if (outfile.is_open()) {
    outfile.write(reinterpret_cast<const char *>(&data.frame_num), sizeof(size_t));
    for (const auto &expr : data.expr) {
      outfile.write(reinterpret_cast<const char *>(expr.data()), expr.size() * sizeof(float));
    }
    for (const auto &jaw : data.jaw_pose) {
      outfile.write(reinterpret_cast<const char *>(jaw.data()), jaw.size() * sizeof(float));
    }
    outfile.close();
  } else {
    LogError("error write file to : {}", filename);
  }
}

AudioToBlendShapeData MnnA2BSSession::Process(int index,
                                              const int16_t *audio_buffer,
                                              size_t length, int sample_rate) {

  if (!audio_to_bs_ || !audio_buffer || length == 0) {
    LogError("Failed to process audio data.");
    return {};
  }
  std::vector<float> float_audio_data;
  for (int i = 0; i < length; i++) {
    float tmp = float(audio_buffer[i]) / 32767.0f;
    float_audio_data.push_back(tmp);
  }
  const size_t chunk_size = sample_rate * 10;
  size_t num_chunks = (float_audio_data.size() + chunk_size - 1) / chunk_size;
  LogDebug("callA2Bs Total Audio Len {}, Chunk Size {}, Num Chunks {}",
           float_audio_data.size(), chunk_size, num_chunks);
  auto begin_time = clk::now();
  std::vector<FLAMEOuput> all_results;
  for (size_t i = 0; i < num_chunks; ++i) {
    LogDebug("callA2Bs begin for {}", i);
    size_t start = i * chunk_size;
    size_t end = std::min(start + chunk_size, float_audio_data.size());
    std::vector<float> current_chunk(float_audio_data.begin() + start,
                                     float_audio_data.begin() + end);
    auto current_result =
        audio_to_bs_->ProcessFLAME(current_chunk, sample_rate);
    all_results.insert(all_results.end(), current_result.begin(),
                       current_result.end());
    auto res = all_results;
    auto end_time = clk::now();
    std::chrono::duration<double> elapsed = end_time - begin_time;
    LogDebug("===> Elapsed time: {} seconds frames count: {}", elapsed.count(),
             res.size());
    if (!res.empty()) {
      LogDebug("===> first frame: {}", res[0].frame_id);
    }
  }
  AudioToBlendShapeData result_data;
  result_data.frame_num = all_results.size();
  for (const auto &result : all_results) {
    result_data.expr.push_back(result.expr);
    result_data.jaw_pose.push_back(result.jaw_pose);
  }
  bs_data_map_[index] = result_data;
  LogDebug("callA2Bs total frame: {}", GetTotalFrameNum());
#if DEBUG_SAVE_A2BS_DATA
  SaveResultDataToBinaryFile(result_data, index);
#endif
  return result_data;
}

AudioToBlendShapeData MnnA2BSSession::Process(int index,
                                              const AudioData &audio_data,
                                              int sample_rate) {
  if (!audio_to_bs_) {
    LogError("Failed to process audio data.");
    return {};
  }
  return Process(index, (int16_t *)audio_data.samples.data(),
                 audio_data.samples.size(), sample_rate);
}

FLAMEOuput MnnA2BSSession::GetActiveFrame(int index, int &segment_index,
                                          int &sub_index) {
  FLAMEOuput flameOuput;
  for (int i = 0; i < bs_data_map_.size(); i++) {
    if (index < bs_data_map_[i].frame_num) {
      segment_index = i;
      sub_index = index;
      flameOuput.expr = bs_data_map_[i].expr[index];
      flameOuput.jaw_pose = bs_data_map_[i].jaw_pose[index];
      flameOuput.frame_id = bs_data_map_[i].frame_num;
      return flameOuput;
    } else {
      index -= bs_data_map_[i].frame_num;
    }
  }
  return flameOuput;
}

size_t MnnA2BSSession::GetTotalFrameNum() {
  int result = 0;
  for (auto i = 0; i < bs_data_map_.size(); i++) {
    if (bs_data_map_.count(i)) {
      result += bs_data_map_[i].frame_num;
    } else {
      break;
    }
  }
  return result;
}

MnnA2BSSession *MnnA2BSSession::GetActiveInstance() { return instance_; }

NAMESPACE_END
