/*
 * @Author: chenjingyu
 * @Date: 2025-08-11 09:15:20
 * @Contact: 2458006466@qq.com
 * @Description: AudioTo3dgsBlendShape
 */
#pragma once

#include "Api.h"

#include "A2bs/AudioToFrameBlendShape.h"
#include "A2bs/GsBodyConverter.h"
#include "Utils/A2BSUtils.h"

NAMESPACE_BEGIN
class API AudioTo3DGSBlendShape {
public:
  AudioTo3DGSBlendShape(const std::string &local_resource_root,
                        const std::string &mnn_mmap_dir, bool do_verts2flame,
                        int ori_fps = 25, int out_fps = 20, int num_exp = 50);
  std::vector<FLAMEOuput> ProcessFLAME(const std::vector<float> &audio,
                                       int sample_rate);

private:
  std::string local_resource_root_;
  AudioToFlameBlendShape atf;
  int num_exp_;
  std::map<int, AudioToFlameBlendShape> _audio_to_blendshape_data;
};

NAMESPACE_END
