/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 19:05:45
 * @Contact: 2458006466@qq.com
 * @Description: ChieseBert
 */
#pragma once

#include "Api.h"

#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>


#include "Utils/Utils.h"

NAMESPACE_BEGIN
using namespace MNN;
using namespace MNN::Express;

class ChineseBert {
public:
  // 为了避免下面的报错，创建一个默认构造函数
  // Constructor for 'MNNTTSSDK' must explicitly initialize the member
  // 'cn_bert_model_' which does not have a default constructor
  ChineseBert();
  ChineseBert(const std::string &local_resource_root,
              const std::string &mnn_mmap_dir);

  // 处理主入口
  std::vector<std::vector<float>> Process(const std::string &text,
                                          const std::vector<int> &word2ph,
                                          const std::string &lang = "zh");

private:
  // 读取BERT的token到id的映射字典
  void ParseBertTokenJsonFile(const std::string &json_path);

  // 保存json为bin
  void SaveBertTokenToBin(const std::string &filename, const bert_token &token);

  // 从bin读取token内容，加速初始化
  void LoadBertTokenFromBin(const std::string &filename, bert_token &token);

  // 将输入的文本转换都bert的token
  // id，注意中英文处理逻辑有所不同，所以需要输入text对应的的语言
  std::vector<int> ObtainBertTokens(const std::string &text,
                                    const std::string &lang = "zh");

private:
  // 资源文件根目录
  std::string resource_root_;

  // token对象
  bert_token bert_token_;

  // MNN 网络相关变量
  int bert_feature_dim_ = 1024;
  std::shared_ptr<Module> module_;
  std::shared_ptr<Executor> executor_;
  std::vector<std::string> input_names{"input_ids", "attention_mask",
                                       "token_type_ids"};
  std::vector<std::string> output_names{"hidden_states"};
};
NAMESPACE_END
