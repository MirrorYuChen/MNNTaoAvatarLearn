cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING "Minimum OS X deployment version. Used only for macOS")

set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

project(ASR LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(INC_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/source)
set(3RD_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdLibs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Build Shared library" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(kaldi-native-fbank)

set(LIBRARIES_TO_LINK "")
list(APPEND LIBRARIES_TO_LINK kaldi-native-fbank-core)

set(MNN_LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdLibs/MNN)
message(STATUS "MNN's dir: ${MNN_LIB_DIR}")
include_directories(${MNN_LIB_DIR}/include)
link_directories(${MNN_LIB_DIR}/lib)
list(APPEND LIBRARIES_TO_LINK MNN)

include_directories(
  ${INC_DIR}
  ${SRC_DIR}
  ${3RD_DIR}/spdlog
  ${3RD_DIR}/nlohmann
)

# src files
file(GLOB_RECURSE SRC_FILES
  ${SRC_DIR}/*.h
  ${SRC_DIR}/*.cc
)
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} FILES ${SRC_FILES})
find_package(Threads REQUIRED)
list(APPEND LIBRARIES_TO_LINK Threads::Threads)

if (BUILD_SHARED_LIBS)
  add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
else (BUILD_SHARED_LIBS)
  add_library(${PROJECT_NAME} STATIC ${SRC_FILES})
endif (BUILD_SHARED_LIBS)
target_link_libraries(${PROJECT_NAME} ${LIBRARIES_TO_LINK})

if (MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  target_compile_options(${PROJECT_NAME} PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor;/Zc:__cplusplus;/utf-8>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/diagnostics:classic>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/wd4251>)
else(MSVC)
  add_definitions(-fPIC -mfma -Wno-unused-function -pedantic)
  target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)
  target_link_libraries(${PROJECT_NAME} dl)
endif (MSVC)

set(TEST_LIST
  TestParseOptions
  TestMnnASRSession
  TestMnnLLMSession
)

foreach(TEST ${TEST_LIST})
  add_executable(${TEST} test/${TEST}.cc)
  target_link_libraries(${TEST} ${PROJECT_NAME})
endforeach(TEST ${TEST_LIST})
