/*
 * @Author: chenjingyu
 * @Date: 2025-08-11 09:07:13
 * @Contact: 2458006466@qq.com
 * @Description: A2BSUtils
 */
#pragma once

#include "Api.h"

#include "MnnA2BSTypes.h"

NAMESPACE_BEGIN
API std::vector<BodyParamsInput>
ReadFramesFromBinary(const std::string &binFilePath, std::string &errorMessage);
API void WriteFramesToBinary(const std::string &binFilePath,
                         const std::vector<BodyParamsInput> &frames);
API std::vector<BodyParamsInput> ParseInputsFromJson(const std::string &json_path);

API std::vector<float> interp_linear(const std::vector<float> &x,
                                 const std::vector<float> &y,
                                 const std::vector<float> &x_new);

API std::vector<std::vector<float>>
resample_bs_params(const std::vector<std::vector<float>> &bs_params, int L2);

API std::vector<std::vector<float>>
convert_to_2d(const std::vector<float> &flat_vector, int N, int M);

API std::vector<float> resampleAudioData(const std::vector<float> &input,
                                     unsigned int sourceSampleRate,
                                     unsigned int targetSampleRate);
API float calculateMean(const std::vector<float> &audio);
API float calculateVariance(const std::vector<float> &audio, float mean);
API std::vector<float> normalizeAudio(std::vector<float> &audio);

NAMESPACE_END
