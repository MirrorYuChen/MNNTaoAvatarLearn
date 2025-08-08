/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 19:05:45
 * @Contact: 2458006466@qq.com
 * @Description: TTSGenerator
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

class TTSGenerator {
public:
  TTSGenerator();
  TTSGenerator(const std::string &tts_generator_model_path,
               const std::string &mnn_mmap_dir);
  std::vector<int16_t> Process(const phone_data &g2p_data_,
                               const std::vector<std::vector<float>> &cn_bert,
                               const std::vector<std::vector<float>> &en_bert);

private:
  // 资源文件根目录
  std::string resource_root_;

  // NOTE 为了保证TTSGenerator类支持复制初始化,
  // 这里不能用unique_ptr，只能用shared_ptr
  std::shared_ptr<Module> module_;
  std::shared_ptr<Executor> executor_;
  std::vector<std::string> input_names{"phone", "tone", "lang_id", "cn_bert",
                                       "en_bert"};
  std::vector<std::string> output_names{"audio"};
  std::shared_ptr<Executor::RuntimeManager> rtmgr_;
  int bert_feature_dim_ = 1024;
};
NAMESPACE_END
