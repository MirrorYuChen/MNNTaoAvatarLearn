 // sherpa-mnn/csrc/bbpe.h
//
// Copyright (c)  2024  Xiaomi Corporation

#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>

// It is equivalent to the map BCHAR_TO_BYTE
// from
// https://github.com/k2-fsa/icefall/blob/master/icefall/byte_utils.py#L280
const std::unordered_map<std::string, uint8_t> &GetByteBpeTable();

