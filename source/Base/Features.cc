   /*
 * @Author: chenjingyu
 * @Date: 2025-08-07 15:10:18
 * @Contact: 2458006366@qq.com
 * @Description: Features
 */
#include "Base/Features.h"

#include <algorithm>
#include <memory>
#include <mutex>  // NOLINT
#include <sstream>
#include <vector>

#include "Base/Resample.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

NAMESPACE_BEGIN
void FeatureExtractorConfig::Register (ParseOptions *po) {
  po->Register("sample-rate", &sampling_rate,
               "Sampling rate of the input waveform. "
               "Note: You can have a different "
               "sample rate for the input waveform. We will do resampling "
               "inside the feature extractor"
  );

  po->Register("feat-dim", &feature_dim,
               "Feature dimension. Must match the one expected by the model. "
               "Not used by whisper and CED models"
  );

  po->Register("low-freq", &low_freq, "Low cutoff frequency for mel bins");

  po->Register("high-freq", &high_freq,
               "High cutoff frequency for mel bins "
               "(if <= 0, offset from Nyquist)"
  );

  po->Register("dither", &dither,
               "Dithering constant (0.0 means no dither). "
               "By default the audio samples are in range [-1,+1], "
               "so 0.00003 is a good value, "
               "equivalent to the default 1.0 from kaldi"
  );
}

std::string FeatureExtractorConfig::ToString () const {
  std::ostringstream os;

  os << "FeatureExtractorConfig(";
  os << "sampling_rate=" << sampling_rate << ", ";
  os << "feature_dim=" << feature_dim << ", ";
  os << "low_freq=" << low_freq << ", ";
  os << "high_freq=" << high_freq << ", ";
  os << "dither=" << dither << ", ";
  os << "normalize_samples=" << (normalize_samples ? "True" : "False") << ", ";
  os << "snip_edges=" << (snip_edges ? "True" : "False") << ")";

  return os.str();
}

class FeatureExtractor::Impl {
public:
  explicit Impl (const FeatureExtractorConfig &config) : config_(config) {
    InitFbank();
  }

  void AcceptWaveform (int32_t sampling_rate, const float *waveform, int32_t n) {
    if (config_.normalize_samples) {
      AcceptWaveformImpl(sampling_rate, waveform, n);
    } else {
      std::vector<float> buf(n);
      for (int32_t i = 0; i != n; ++i) {
        buf[i] = waveform[i] * 32768;
      }
      AcceptWaveformImpl(sampling_rate, buf.data(), n);
    }
  }

  void AcceptWaveformImpl (int32_t sampling_rate, const float *waveform,
                           int32_t n) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (resampler_) {
      if (sampling_rate != resampler_->GetInputSamplingRate()) {
        LogError(
          "You changed the input sampling rate!! Expected: {}, given: {}",
          resampler_->GetInputSamplingRate(), sampling_rate
        );
        exit(-1);
      }

      std::vector<float> samples;
      resampler_->Resample(waveform, n, false, &samples);
      fbank_->AcceptWaveform(config_.sampling_rate, samples.data(),
                             samples.size()
      );
      return;
    }

    if (sampling_rate != config_.sampling_rate) {
      LogError(
        "Creating a resampler: in_sample_rate: {}, output_sample_rate: {}\n",
        sampling_rate, static_cast<int32_t>(config_.sampling_rate)
      );

      float min_freq = std::min<int32_t>(sampling_rate, config_.sampling_rate);
      float lowpass_cutoff = 0.99 * 0.5 * min_freq;

      int32_t lowpass_filter_width = 6;
      resampler_ = std::make_unique<LinearResample>(
        sampling_rate, config_.sampling_rate, lowpass_cutoff,
        lowpass_filter_width
      );

      std::vector<float> samples;
      resampler_->Resample(waveform, n, false, &samples);
      fbank_->AcceptWaveform(config_.sampling_rate, samples.data(),
                             samples.size()
      );
      return;
    }

    fbank_->AcceptWaveform(sampling_rate, waveform, n);
  }

  void InputFinished () const {
    std::lock_guard<std::mutex> lock(mutex_);
    fbank_->InputFinished();
  }

  int32_t NumFramesReady () const {
    std::lock_guard<std::mutex> lock(mutex_);
    return fbank_->NumFramesReady();
  }

  bool IsLastFrame (int32_t frame) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return fbank_->IsLastFrame(frame);
  }

  std::vector<float> GetFrames (int32_t frame_index, int32_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frame_index + n > fbank_->NumFramesReady()) {
      LogError("{} + {} > {}\n", frame_index, n,
                       fbank_->NumFramesReady()
      );
      exit(-1);
    }

    int32_t discard_num = frame_index - last_frame_index_;
    if (discard_num < 0) {
      LogError("last_frame_index_: {}, frame_index_: {}",
                       last_frame_index_, frame_index
      );
      exit(-1);
    }
    fbank_->Pop(discard_num);

    int32_t feature_dim = fbank_->Dim();
    std::vector<float> features(feature_dim * n);

    float *p = features.data();

    for (int32_t i = 0; i != n; ++i) {
      const float *f = fbank_->GetFrame(i + frame_index);
      std::copy(f, f + feature_dim, p);
      p += feature_dim;
    }

    last_frame_index_ = frame_index;

    return features;
  }

  int32_t FeatureDim () const {
    return opts_.mel_opts.num_bins;
  }

private:
  void InitFbank () {
    opts_.frame_opts.dither = config_.dither;
    opts_.frame_opts.snip_edges = config_.snip_edges;
    opts_.frame_opts.samp_freq = config_.sampling_rate;
    opts_.frame_opts.frame_shift_ms = config_.frame_shift_ms;
    opts_.frame_opts.frame_length_ms = config_.frame_length_ms;
    opts_.frame_opts.remove_dc_offset = config_.remove_dc_offset;
    opts_.frame_opts.preemph_coeff = config_.preemph_coeff;
    opts_.frame_opts.window_type = config_.window_type;

    opts_.mel_opts.num_bins = config_.feature_dim;

    opts_.mel_opts.high_freq = config_.high_freq;
    opts_.mel_opts.low_freq = config_.low_freq;

    opts_.mel_opts.is_librosa = config_.is_librosa;

    fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
  }

private:
  std::unique_ptr<knf::OnlineFbank> fbank_;
  knf::FbankOptions opts_;
  FeatureExtractorConfig config_;
  mutable std::mutex mutex_;
  std::unique_ptr<LinearResample> resampler_;
  int32_t last_frame_index_ = 0;
};

FeatureExtractor::FeatureExtractor (const FeatureExtractorConfig &config /*={}*/)
  : impl_(std::make_unique<Impl>(config)) {
}

FeatureExtractor::~FeatureExtractor () = default;

void FeatureExtractor::AcceptWaveform (int32_t sampling_rate,
                                       const float *waveform, int32_t n) const {
  impl_->AcceptWaveform(sampling_rate, waveform, n);
}

void FeatureExtractor::InputFinished () const { impl_->InputFinished(); }

int32_t FeatureExtractor::NumFramesReady () const {
  return impl_->NumFramesReady();
}

bool FeatureExtractor::IsLastFrame (int32_t frame) const {
  return impl_->IsLastFrame(frame);
}

std::vector<float> FeatureExtractor::GetFrames (int32_t frame_index,
                                                int32_t n) const {
  return impl_->GetFrames(frame_index, n);
}

int32_t FeatureExtractor::FeatureDim () const { return impl_->FeatureDim(); }
NAMESPACE_END
