  /*
 * @Author: chenjingyu
 * @Date: 2025-08-08 13:09:45
 * @Contact: 2458006366@qq.com
 * @Description: WaveReader
 */
#pragma once

#include "Api.h"
#include <istream>
#include <string>
#include <vector>

NAMESPACE_BEGIN
/** Read a wave file with expected sample rate.

    @param filename Path to a wave file. It MUST be single channel, 16-bit
                    PCM encoded.
    @param sampling_rate  On return, it contains the sampling rate of the file.
    @param is_ok On return it is true if the reading succeeded; false otherwise.

    @return Return wave samples normalized to the range [-1, 1).
 */
API std::vector<float> ReadWave (const std::string &filename, int32_t *sampling_rate,
                                 bool *is_ok);

API std::vector<float> ReadWave (std::istream &is, int32_t *sampling_rate,
                                 bool *is_ok);

NAMESPACE_END
