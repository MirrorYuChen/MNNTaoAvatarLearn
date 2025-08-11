/*
 * @Author: chenjingyu
 * @Date: 2025-08-11 09:05:37
 * @Contact: 2458006466@qq.com
 * @Description: A2bsTypes
 */
#pragma once

#include "Api.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include <json.hpp>

NAMESPACE_BEGIN
using json = nlohmann::json;

struct API FLAMEOuput {
  bool IsEmpty() const { return expr.empty(); }
  void Reset() {
    frame_id = -1;
    expr.clear();
    jaw_pose.clear();
  }

  int frame_id;
  std::vector<float> expr;
  std::vector<float> jaw_pose;
};

struct BodyParamsInput {
  int frame_id;
  std::vector<float> expression;
  std::vector<float> Rh;
  std::vector<float> Th;
  std::vector<float> body_pose;
  std::vector<float> jaw_pose;
  std::vector<float> leye_pose;
  std::vector<float> reye_pose;
  std::vector<float> left_hand_pose;
  std::vector<float> right_hand_pose;
  std::vector<float> eye_verts;
  std::vector<float> pose;
};

struct FullTypeOutput {
  int frame_id;
  std::vector<float> expr;
  std::vector<float> joints_transform;
  std::vector<float> local_joints_transform;
  std::vector<float> pose_z;
  std::vector<float> app_pose_z;
  std::vector<float> pose;
};

struct API AudioToBlendShapeData {
  // per frame's data
  std::vector<std::vector<float>> expr;
  std::vector<std::vector<float>> pose;
  std::vector<std::vector<float>> pose_z;
  std::vector<std::vector<float>> app_pose_z;
  std::vector<std::vector<float>> jaw_pose;
  std::vector<std::vector<float>> joints_transform;
  size_t frame_num = 0;
};

struct API AudioData {
  std::vector<int16_t> samples;          // PCM样本数据
  std::vector<float> normalized_samples; // 归一化到[-1, 1]的PCM样本数据
};

NAMESPACE_END
