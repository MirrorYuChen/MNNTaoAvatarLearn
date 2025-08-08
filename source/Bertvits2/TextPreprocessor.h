/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 19:05:45
 * @Contact: 2458006466@qq.com
 * @Description: TextNormalizer
 */
#pragma once

#include "Api.h"
#include "Utils/Utils.h"

NAMESPACE_BEGIN
using json = nlohmann::json;

class TextNormalizer {
public:
  TextNormalizer();

  std::vector<std::string> Process(const std::string &text);

  // 从原始句子中获取中文的部分
  std::vector<std::string> SplitCnPart(const std::string &text);

  // 从原始句子中获取英文的部分
  std::vector<std::string> SplitEnPart(const std::string &text);

  // 对句子进行规范化，包括繁体转简体、全角转半角、日期、时间转换，电话号码转换等
  std::string NormalizeSentence(const std::string &sentence);

private:
  // 去除句子中的特殊中文标点符号，如书名号，各种括号，引号等
  std::string RemoveSpecialChineseSymbols(const std::string &text);

private:
  std::regex SENTENCE_SPLITOR;
};

class TextPreprocessor {
public:
  TextPreprocessor();
  std::vector<std::vector<SentLangPair>> Process(std::string &text,
                                                 int split_max_len = 1);

  // 替换自定义的替换组合，如B2C->B2C
  std::string ReplaceSpecialText(std::string &text);

  // 分离字符串，返回段落和语言的对
  std::vector<std::pair<std::string, std::string>>
  SplitByLang(const std::string &text, const std::string &digit_type = "");

  // 合并相邻的中文/英文segments
  std::vector<std::pair<std::string, std::string>> MergeAdjacent(
      const std::vector<std::pair<std::string, std::string>> &lang_splits);

private:
  TextNormalizer normalizer;
  std::string SENTENCE_SPLITOR = "：。？！?!";
};

NAMESPACE_END
