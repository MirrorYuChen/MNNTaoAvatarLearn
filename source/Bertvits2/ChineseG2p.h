/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 19:05:45
 * @Contact: 2458006466@qq.com
 * @Description: ChineseG2p
 */
#pragma once

#include "Api.h"

#include "Bertvits2/AnToCn.h"
#include "Bertvits2/Pinyin.h"
#include "Bertvits2/ToneAdjuster.h"

NAMESPACE_BEGIN
class ChineseG2P {
public:
  ChineseG2P(const std::string &local_resource_root);

  // 处理主入口
  std::tuple<std::string, phone_data> Process(SentLangPair &text);

private:
  // g2p实现入口，进行分句前处理,调用G2pInternal，然后数据后处理
  g2p_data G2P(std::string &text);

  // g2p具体实现
  g2p_data G2PInternal(const std::vector<std::string> &text);

  // 读取json文件
  void ParsePinyToSymbolJsonFile(const std::string &pinyin_json_path);

  // 读取用户自定义中文热词列表
  void ParseHotwordsCNFile(const std::string &hotwords_cn_json_path);

  // 将热词写入到二进制文件，加速读取
  void SaveHotwordsToBin(const std::string &filename);

  // 从二进制文件读取热词
  void LoadHotwordsFromBin(const std::string &filename);

  // 将声母和韵母转换为音素和音调
  g2p_data
  ConvertInitialFinalToPhoneTone(const std::vector<std::string> &initials,
                                 const std::vector<std::string> &finals);

  // 将音素和音调转换为index，同时进行差值
  phone_data CleanedTextToSequence(const g2p_data &g2p_data_,
                                   const std::string &language);

  // 替换中文标点为英文标点
  std::string ReplacePunctuations(const std::string &text);

  // 针对中文的进一步规范化操作，主要就是调用 an2cn和标点符号替换
  std::string TextNormalize(const std::string &text);

  // 阿拉伯数字转中文汉字，如'123.58'-> '一百二十三点五八'
  std::string TextAn2cn(std::string &text);

private:
  // 资源文件根目录
  std::string resource_root_;

  Pinyin pinyin_;
  ToneAdjuster tone_adjuster_;
  WordSpliter &word_spliter_;
  An2Cn an2cn_;

  // 拼音到符号的映射
  pinyin_to_symbol_map pinyin_to_symbol_map_;
  std::unordered_map<std::string, int> _symbol_to_id;
  std::map<std::string,
           std::tuple<std::vector<std::string>, std::vector<std::string>>>
      custom_phone_;
};

NAMESPACE_END
