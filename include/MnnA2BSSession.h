/*
 * @Author: chenjingyu
 * @Date: 2025-08-11 09:26:35
 * @Contact: 2458006466@qq.com
 * @Description: MnnA2BSSession
 */
#pragma once

#include "Api.h"
#include "MnnA2BSTypes.h"

NAMESPACE_BEGIN
class AudioTo3DGSBlendShape;
class API MnnA2BSSession {
public:
  MnnA2BSSession();
  ~MnnA2BSSession();
  static MnnA2BSSession *GetActiveInstance();

  bool LoadA2bsResources(const char *res_path, const char *temp_path);
  void SaveResultDataToBinaryFile(const AudioToBlendShapeData &data, int index);
  AudioToBlendShapeData Process(int index, const int16_t *audio_buffer,
                                size_t length, int sample_rate);
  AudioToBlendShapeData Process(int index, const AudioData &audio_data,
                                int sample_rate);
  FLAMEOuput GetActiveFrame(int index, int &segment_index, int &sub_index);
  size_t GetTotalFrameNum();

private:
  std::shared_ptr<AudioTo3DGSBlendShape> audio_to_bs_ = nullptr;
  std::map<int, AudioToBlendShapeData> bs_data_map_;
  static MnnA2BSSession *instance_;
};

NAMESPACE_END
