/*
 * @Author: chenjingyu
 * @Date: 2024-05-27 14:35:47
 * @Contact: 2458006466@qq.com
 * @Description: Api
 */
#pragma once

#include "Platform.h"

#define EXTERN_C extern "C"

#if OS_WINDOWS
#define API_IMPORT __declspec(dllimport)
#define API_EXPORT __declspec(dllexport)
#define API_LOCAL
#else
#if defined(__GNUC__) && __GNUC__ >= 4
#define API_IMPORT __attribute__((visibility("default")))
#define API_EXPORT __attribute__((visibility("default")))
#define API_LOCAL  __attribute__((visibility("hidden")))
#else
#define API_IMPORT
#define API_EXPORT
#define API_LOCAL
#endif
#endif

#define API API_EXPORT

#ifdef __cplusplus
#define C_API EXTERN_C API
#else
#define C_API API
#endif

#define NAMESPACE mirror

#define NAMESPACE_BEGIN namespace NAMESPACE {
#define NAMESPACE_END   }  // namespace mirror

NAMESPACE_BEGIN
#ifndef M_PI
#define M_PI 3.14159265358979323846  // pi
#endif

#define MAX_(x, y) ((x) > (y) ? (x) : y)
#define MIN_(x, y) ((x) < (y) ? (x) : y)

#define NOT_ALLOWED_COPY(Type)                                                                                         \
  Type(const Type &)  = delete;                                                                                        \
  const Type &operator=(const Type &) = delete;

#define NOT_ALLOWED_MOVE(Type)                                                                                         \
  Type(const Type &) = delete;                                                                                         \
  Type &operator=(const Type &) = delete;                                                                              \
  Type(Type &&) noexcept        = delete;                                                                              \
  Type &operator=(Type &&) noexcept = delete;


#define STRING_CONCAT_HELPER(a, b, c) a##b##_##c
#define STRING_CONCAT(a, b, c)        STRING_CONCAT_HELPER(a, b, c)

#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x)        STRINGIFY_HELPER(x)

#define TYPE_MAP_OBJECT_NAME STRING_CONCAT(obj_, __LINE__, __COUNTER__)
#define FILE_LINE            __FILE__ ":line_" STRINGIFY(__LINE__)

NAMESPACE_END
