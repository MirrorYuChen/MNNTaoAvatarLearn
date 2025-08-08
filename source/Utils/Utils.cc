/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 12:10:28
 * @Contact: 2458006366@qq.com
 * @Description: Utils
 */
#include "Utils/Utils.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Base/Logger.h"
#include "Utils/TextUtils.h"

NAMESPACE_BEGIN
static bool EncodeBase (const std::vector<std::string> &lines,
                        const SymbolTable &symbol_table,
                        std::vector<std::vector<int32_t> > *ids,
                        std::vector<std::string> *phrases,
                        std::vector<float> *scores,
                        std::vector<float> *thresholds) {
  ids->clear();

  std::vector<int32_t> tmp_ids;
  std::vector<float> tmp_scores;
  std::vector<float> tmp_thresholds;
  std::vector<std::string> tmp_phrases;

  std::string word;
  bool has_scores = false;
  bool has_thresholds = false;
  bool has_phrases = false;
  bool has_oov = false;

  for (const auto &line: lines) {
    float score = 0;
    float threshold = 0;
    std::string phrase = "";

    std::istringstream iss(line);
    while (iss >> word) {
      if (symbol_table.Contains(word)) {
        int32_t id = symbol_table[word];
        tmp_ids.push_back(id);
      } else {
        switch (word[0]) {
          case ':' : // boosting score for current keyword
            score = std::stof(word.substr(1));
            has_scores = true;
            break;
          case '#' : // triggering threshold (probability) for current keyword
            threshold = std::stof(word.substr(1));
            has_thresholds = true;
            break;
          case '@' : // the original keyword string
            phrase = word.substr(1);
            has_phrases = true;
            break;
          default :
            LogError(
              "Cannot find ID for token {} at line: {}. (Hint: Check the "
              "tokens.txt see if {} in it)",
              word, line, word
            );
            has_oov = true;
            break;
        }
      }
    }
    ids->push_back(std::move(tmp_ids));
    tmp_ids = {};
    tmp_scores.push_back(score);
    tmp_phrases.push_back(phrase);
    tmp_thresholds.push_back(threshold);
  }
  if (scores != nullptr) {
    if (has_scores) {
      scores->swap(tmp_scores);
    } else {
      scores->clear();
    }
  }
  if (phrases != nullptr) {
    if (has_phrases) {
      *phrases = std::move(tmp_phrases);
    } else {
      phrases->clear();
    }
  }
  if (thresholds != nullptr) {
    if (has_thresholds) {
      thresholds->swap(tmp_thresholds);
    } else {
      thresholds->clear();
    }
  }
  return !has_oov;
}

bool EncodeKeywords (std::istream &is, const SymbolTable &symbol_table,
                     std::vector<std::vector<int32_t> > *keywords_id,
                     std::vector<std::string> *keywords,
                     std::vector<float> *boost_scores,
                     std::vector<float> *threshold) {
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(is, line)) {
    lines.push_back(line);
  }
  return EncodeBase(lines, symbol_table, keywords_id, keywords, boost_scores,
                    threshold
  );
}

NAMESPACE_END
