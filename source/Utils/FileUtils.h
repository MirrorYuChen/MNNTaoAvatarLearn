   /*
 * @Author: chenjingyu
 * @Date: 2025-08-07 14:27:10
 * @Contact: 2458006366@qq.com
 * @Description: FileUtils
 */
#pragma once

#include "Api.h"

#include <fstream>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

NAMESPACE_BEGIN
/** Check whether a given path is a file or not
 *
 * @param filename Path to check.
 * @return Return true if the given path is a file; return false otherwise.
 */
bool FileExists(const std::string &filename);

/** Abort if the file does not exist.
 *
 * @param filename The file to check.
 */
void AssertFileExists(const std::string &filename);

std::vector<char> ReadFile(const std::string &filename);

#if __ANDROID_API__ >= 9
std::vector<char> ReadFile(AAssetManager *mgr, const std::string &filename);
#endif

#if __OHOS__
std::vector<char> ReadFile(NativeResourceManager *mgr,
                           const std::string &filename);
#endif

NAMESPACE_END
