/*
 * @Author: chenjingyu
 * @Date: 2024-05-27 14:35:42
 * @Contact: 2458006466@qq.com
 * @Description: Platform
 */
#pragma once

#include <string>

#if defined(__ANDROID__)
#define OS_ANDROID 1
#define OS_WINDOWS 0
#define OS_MAC 0
#define OS_LINUX 1
#define OS_IOS 0
#elif defined(__WINDOWS__) || defined(_WIN32) || defined(WIN32) ||             \
    defined(_WIN64) || defined(WIN64) || defined(__WIN32__) ||                 \
    defined(__TOS_WIN__)
#define OS_ANDROID 0
#define OS_WINDOWS 1
#define OS_MAC 0
#define OS_LINUX 0
#define OS_IOS 0
#elif defined(__MACOSX) || defined(__MACOS_CLASSIC__) || defined(__APPLE__) || \
    defined(__apple__)
#include "TargetConditionals.h"
#if TARGET_IPHONE_SIMULATOR || TARGET_OS_IPHONE
#define OS_ANDROID 0
#define OS_WINDOWS 0
#define OS_MAC 0
#define OS_LINUX 0
#define OS_IOS 1
#elif TARGET_OS_MAC
#define OS_ANDROID 0
#define OS_WINDOWS 0
#define OS_MAC 1
#define OS_LINUX 0
#define OS_IOS 0
#else
//#   error "Unknown Apple platform"
#define OS_ANDROID 0
#define OS_WINDOWS 0
#define OS_MAC 0
#define OS_LINUX 0
#define OS_IOS 0
#endif
#elif defined(__linux__) || defined(linux) || defined(__linux) ||              \
    defined(__LINUX__) || defined(LINUX) || defined(_LINUX)
#define OS_ANDROID 0
#define OS_WINDOWS 0
#define OS_MAC 0
#define OS_LINUX 1
#define OS_IOS 0
#else
//#   error Unknown OS
#define OS_ANDROID 0
#define OS_WINDOWS 0
#define OS_MAC 0
#define OS_LINUX 0
#define OS_IOS 0

#endif

#ifndef HAS_RTTI
// Detect if RTTI is disabled in the compiler.
#if defined(__clang__) && defined(__has_feature)
#define HAS_RTTI __has_feature(cxx_rtti)
#elif defined(__GNUC__) && !defined(__GXX_RTTI)
#define HAS_RTTI 0
#elif defined(_MSC_VER) && !defined(_CPPRTTI)
#define HAS_RTTI 0
#else
#define HAS_RTTI 1
#endif
#endif  // HAS_RTTI
