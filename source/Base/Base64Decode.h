/*
 * @Author: chenjingyu
 * @Date: 2025-08-06 16:28:08
 * @Contact: 2458006366@qq.com
 * @Description: Base64Decode
 */
#pragma once

#include "Api.h"
#include <string>

NAMESPACE_BEGIN
/** @param s A base64 encoded string.
 *  @return Return the decoded string.
 */
std::string Base64Decode (const std::string &s);
NAMESPACE_END
