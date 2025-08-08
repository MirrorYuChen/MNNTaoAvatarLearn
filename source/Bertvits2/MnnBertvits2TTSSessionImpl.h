/*
 * @Author: chenjingyu
 * @Date: 2025-08-08 19:05:45
 * @Contact: 2458006466@qq.com
 * @Description:MnnBertvits2TTSSessionImpl
 */
#pragma once
#include "Api.h"
#include "MnnTTSSession.h"
#include "Bertvits2/ChineseBert.h"
#include "Bertvits2/ChineseG2p.h"
#include "Bertvits2/EnglishBert.h"
#include "Bertvits2/EnglishG2p.h"
#include "Bertvits2/TTSGenerator.h"
#include "Bertvits2/TextPreprocessor.h"
#include "Utils/Utils.h"


NAMESPACE_BEGIN
using json = nlohmann::json;

typedef std::vector<int16_t> Audio;

class MnnBertvits2TTSSessionImpl : public MnnTTSSessionImplBase {
public:
  MnnBertvits2TTSSessionImpl(const std::string &local_resource_root,
                             const std::string &tts_generator_model_path,
                             const std::string &mnn_mmap_dir);

  std::tuple<int, Audio> Process(const std::string &text) override;

private:
  std::tuple<phone_data, std::vector<std::vector<float>>,
             std::vector<std::vector<float>>>
  ExtractPhoneTextFeatures(const std::vector<SentLangPair> &word_list_by_lang);

private:
  std::string resource_root_;
  int sample_rate_ = 44100;

  TextPreprocessor text_preprocessor_;
  ChineseG2P cn_g2p_;
  EnglishG2P en_g2p_;
  ChineseBert cn_bert_model_;
  EnglishBert en_bert_model_;
  TTSGenerator tts_generator_;
};

NAMESPACE_END
