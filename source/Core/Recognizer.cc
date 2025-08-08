  /*
 * @Author: chenjingyu
 * @Date: 2025-08-08 09:51:09
 * @Contact: 2458006366@qq.com
 * @Description: Recognizer
 */
#include "Core/Recognizer.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "Core/RecognizerImpl.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "Utils/FileUtils.h"
#include "Utils/TextUtils.h"

NAMESPACE_BEGIN
Recognizer::Recognizer (const RecognizerConfig &config)
  : impl_(RecognizerImpl::Create(config)) {
}

Recognizer::~Recognizer () = default;

std::unique_ptr<Stream> Recognizer::CreateStream () const {
  return impl_->CreateStream();
}

bool Recognizer::IsReady (Stream *s) const {
  return impl_->IsReady(s);
}

void Recognizer::DecodeStreams (Stream **ss, int32_t n) const {
  impl_->DecodeStreams(ss, n);
}

RecognizerResult Recognizer::GetResult (Stream *s) const {
  return impl_->GetResult(s);
}

bool Recognizer::IsEndpoint (Stream *s) const {
  return impl_->IsEndpoint(s);
}

void Recognizer::Reset (Stream *s) const { impl_->Reset(s); }

NAMESPACE_END
