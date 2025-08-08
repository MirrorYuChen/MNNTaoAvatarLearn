 /*
 * @Author: chenjingyu
 * @Date: 2025-08-07 17:05:52
 * @Contact: 2458006366@qq.com
 * @Description: Model
 */
#pragma once

#include "Api.h"
#include "Base/RecognizerConfig.h"
#include "Core/Hypothesis.h"
#include "Core/MNNUtils.h"

NAMESPACE_BEGIN
struct API DecoderResult {
  /// Number of frames after subsampling we have decoded so far
  int32_t frame_offset = 0;

  /// The decoded token IDs so far
  std::vector<int> tokens;

  /// number of trailing blank frames decoded so far
  int32_t num_trailing_blanks = 0;

  /// timestamps[i] contains the output frame index where tokens[i] is decoded.
  std::vector<int32_t> timestamps;
  std::vector<float> ys_probs;

  // Cache decoder_out for endpointing
  MNN::Express::VARP decoder_out;

  // used only in modified beam_search
  Hypotheses hyps;

  DecoderResult()
      : tokens{}, num_trailing_blanks(0), decoder_out{nullptr}, hyps{} {}

  DecoderResult(const DecoderResult &other);

  DecoderResult &operator=(const DecoderResult &other);

  DecoderResult(DecoderResult &&other) noexcept;

  DecoderResult &operator=(DecoderResult &&other) noexcept;
};

class API Model {
public:
  explicit Model(const ModelConfig &cfg);
  ~Model();

  static std::unique_ptr<Model> Create(const ModelConfig &cfg);

  [[nodiscard]] std::vector<MNN::Express::VARP>
  StackStates(const std::vector<std::vector<MNN::Express::VARP>> &states) const;

  [[nodiscard]] std::vector<std::vector<MNN::Express::VARP>>
  UnStackStates(const std::vector<MNN::Express::VARP> &states) const;

  std::vector<MNN::Express::VARP> GetEncoderInitStates();

  std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>>
  RunEncoder(MNN::Express::VARP features,
             std::vector<MNN::Express::VARP> states,
             MNN::Express::VARP processed_frames);

  MNN::Express::VARP RunDecoder(MNN::Express::VARP decoder_input);

  MNN::Express::VARP RunJoiner(MNN::Express::VARP encoder_out,
                               MNN::Express::VARP decoder_out);

  [[nodiscard]] int32_t ContextSize() const { return context_size_; }

  [[nodiscard]] int32_t ChunkSize() const { return T_; }

  [[nodiscard]] int32_t ChunkShift() const { return decode_chunk_len_; }

  [[nodiscard]] int32_t VocabSize() const { return vocab_size_; }
  [[nodiscard]] MNNAllocator *Allocator() const { return allocator_; }

private:
  void InitEncoder(void *model_data, size_t model_data_length);

  void InitDecoder(void *model_data, size_t model_data_length);

  void InitJoiner(void *model_data, size_t model_data_length);

private:
  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator *allocator_;

  std::unique_ptr<MNN::Express::Module> encoder_sess_;
  std::unique_ptr<MNN::Express::Module> decoder_sess_;
  std::unique_ptr<MNN::Express::Module> joiner_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;

  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  std::vector<std::string> joiner_input_names_;
  std::vector<const char *> joiner_input_names_ptr_;

  std::vector<std::string> joiner_output_names_;
  std::vector<const char *> joiner_output_names_ptr_;

  ModelConfig cfg_;

  std::vector<int32_t> encoder_dims_;
  std::vector<int32_t> attention_dims_;
  std::vector<int32_t> num_encoder_layers_;
  std::vector<int32_t> cnn_module_kernels_;
  std::vector<int32_t> left_context_len_;

  int32_t T_ = 0;
  int32_t decode_chunk_len_ = 0;

  int32_t context_size_ = 0;
  int32_t vocab_size_ = 0;
};

static MNN::Express::VARP
BuildDecoderInput(Model *model, const std::vector<DecoderResult> &results) {
  int32_t batch_size = static_cast<int32_t>(results.size());
  int32_t context_size = model->ContextSize();
  std::array<int, 2> shape{batch_size, context_size};
  MNN::Express::VARP decoder_input =
      MNNUtilsCreateTensor<int>(model->Allocator(), shape.data(), shape.size());
  int *p = decoder_input->writeMap<int>();

  for (const auto &r : results) {
    const int *begin = r.tokens.data() + r.tokens.size() - context_size;
    const int *end = r.tokens.data() + r.tokens.size();
    std::copy(begin, end, p);
    p += context_size;
  }
  return decoder_input;
}

static MNN::Express::VARP
BuildDecoderInput(Model *model, const std::vector<Hypothesis> &hyps) {
  int32_t batch_size = static_cast<int32_t>(hyps.size());
  int32_t context_size = model->ContextSize();
  std::array<int, 2> shape{batch_size, context_size};
  MNN::Express::VARP decoder_input =
      MNNUtilsCreateTensor<int>(model->Allocator(), shape.data(), shape.size());
  int *p = decoder_input->writeMap<int>();

  for (const auto &h : hyps) {
    std::copy(h.ys.end() - context_size, h.ys.end(), p);
    p += context_size;
  }
  return decoder_input;
}

NAMESPACE_END
