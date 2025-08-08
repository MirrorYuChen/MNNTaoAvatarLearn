/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 19:05:45
 * @Contact: 2458006466@qq.com
 * @Description: EnglishBert
 */
#pragma once

#include "Api.h"
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

#include "Utils/Utils.h"

NAMESPACE_BEGIN
namespace fs = std::filesystem;

using namespace MNN;
using namespace MNN::Express;

class EnglishBert {
public:
  EnglishBert();
  EnglishBert(const std::string &local_resource_root);
  std::vector<std::vector<float>> Process(const std::string &text,
                                          const std::vector<int> &word2ph);

private:
  std::vector<int> ObtainBertTokens(const std::string &text);
  void ParseBertTokenJsonFile(const std::string &json_path);

private:
  std::string local_resource_root_;

  // tokenizer
  // BertTokenizer bert_tokenizer_;

  bert_token bert_token_;

  // MNN 网络相关变量
  int bert_feature_dim_ = 1024;
  std::shared_ptr<Module> module; // module
  std::vector<std::string> input_names{"input_ids", "token_type_ids",
                                       "attention_mask"};
  std::vector<std::string> output_names{"hidden_states"};
};
NAMESPACE_END
