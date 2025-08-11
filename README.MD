## MNNTaoAvatarLearn
* MNN团队开源的[MNNTaoAvatar](https://github.com/alibaba/MNN/tree/master/apps/Android/MnnTaoAvatar)学习笔记
* Author: Chenjingyu
* Date: 20250808

## 1.项目简介
* MNNTaoAvatar包含五个部分：
  - MNN-LLM(本地聊天机器人)：基于本地运行的LLM，实时与数字人畅聊
  - Sherpa-MNN-ASR(语音识别更智能)：内置ASR模型，即说即转文字
  - MNN-TTS(随心所欲合成语音)：TTS模型，让你的数字人发声自然真实
  - MNN-A2BS(声音驱动表情动作)：A2BS技术，通过声音自动生成数字人丰富的面部表情和动作
  - MNN-NNR(实时神经渲染)：让数字人表情细腻逼真，互动感更强

&emsp;目前移植了四个模块：MNN-LLM对应MnnLLMSession，Sharpa-MNN-ASR对应MnnASRSession和MNN-TTS对应MnnTTSSession，MNN-A2BS对应MnnA2BSSession，其中ASR模块是对原始[sherpa-mnn](https://github.com/alibaba/MNN/tree/master/apps/frameworks/sherpa-mnn)裁剪后的版本，这个比较费事，其余三个是直接抄过来的，主要是做了一些代码整理，A2BS结果是否正确需要移植NNR模块，然而NNR没有开源，所以没有往下整的必要，这个就先这样吧。


## 2.使用方法
&emsp;先看一下模型存在项目结构
```bash
.
├── 3rdLibs
│   ├── MNN
│   │   ├── include
│   │   └── lib
│   ├── nlohmann
│   │   └── json.hpp
│   └── spdlog
├── CMakeLists.txt
├── README.MD
├── build
├── cmake
│   └── kaldi-native-fbank.cmake
├── data
│   ├── a2bs
│   │   ├── README.md
│   │   ├── audio2verts.mnn
│   │   ├── body_converter.mnn
│   │   ├── body_params.bin
│   │   ├── configuration.json
│   │   └── idle_speech_slices.json
│   ├── asr
│   │   ├── 1.wav
│   │   ├── README.md
│   │   ├── configuration.json
│   │   ├── decoder.mnn
│   │   ├── encoder.mnn
│   │   ├── joiner.mnn
│   │   └── tokens.txt
│   ├── llm
│   │   ├── README.md
│   │   ├── config.json
│   │   ├── configuration.json
│   │   ├── llm.mnn
│   │   ├── llm.mnn.json
│   │   ├── llm.mnn.weight
│   │   ├── llm_config.json
│   │   └── tokenizer.txt
│   └── tts
│       ├── 38acd89e9b396e6b
│       ├── b4da26028007a684
│       ├── common
│       │   ├── mnn_models
│       │   │   ├── chinese_bert.mnn
│       │   │   ├── chinese_bert.mnn.weight
│       │   │   ├── english_bert.mnn
│       │   │   └── english_bert.mnn.weight
│       │   └── text_processing_jsons
│       │       ├── char_state.bin
│       │       ├── cn_bert_token.bin
│       │       ├── default_tone_words.json
│       │       ├── en_bert_token.json
│       │       ├── eng_dict.bin
│       │       ├── hotwords_cn.bin
│       │       ├── hotwords_cn.json
│       │       ├── phrases_dict.bin
│       │       ├── pinyin_dict.bin
│       │       ├── pinyin_to_symbol_map.bin
│       │       ├── prob_emit.bin
│       │       ├── prob_start.bin
│       │       ├── prob_trans.bin
│       │       ├── tokenizer.txt
│       │       ├── word_freq.bin
│       │       └── word_tag.bin
│       ├── config.json
│       ├── configuration.json
│       ├── tokenizer.txt
│       └── tts_generator_w_bert_chenxi_0310_int8.mnn
├── include
├── source
└── test
```
&emsp;这里主要看一下模型存放的目录结构。
* (1) 编译最新版的MNN
```bash
>> git clone git@github.com:alibaba/MNN.git
>> cd MNN && mkdir build && cd build
>> cmake .. -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true         \
-DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_SEP_BUILD=OFF \
-DMNN_BUILD_CONVERTER=ON -DCMAKE_INSTALL_PREFIX=.
>> make install
```
* (2) 将编译好的lib和include替换掉`3rdLibs/MNN`下面对应文件
* (3) 编译本项目
```bash
>> mkdir build && cd build && cmake .. && make -j 16
```
* (4) 测试效果

&emsp;所有MNNTaoAvatar的模型都可以在这个地址找到：https://modelscope.cn/collections/TaoAvatar-68d8a46f2e554a
  - (a) 测试MNN-TTS模块：先下载模型，对应地址：[bert-vits2-MNN](https://modelscope.cn/models/MNN/bert-vits2-MNN)，我这里将其解压之后，放到了项目底下的data/tts路径，具体看代码
```bash
>> ./TestMnnTTSSession
[2025-08-08 20:32:18.785] [mirror] [info] [Pinyin.cc:64] Pinyin 开始初始化...
[2025-08-08 20:32:18.841] [mirror] [info] [Pinyin.cc:96] Pinyin 初始化成功, timecost: 55ms
[2025-08-08 20:32:18.841] [mirror] [info] [WordSpliter.cc:43] WordSpliter 开始初始化...
[2025-08-08 20:32:19.154] [mirror] [info] [WordSpliter.cc:55] WordSpliter 初始化完成, timecost: 313ms
[2025-08-08 20:32:19.154] [mirror] [info] [ToneAdjuster.cc:14] ToneAdjuster 开始初始化...
[2025-08-08 20:32:19.154] [mirror] [info] [ToneAdjuster.cc:36]  ToneAdjuster 初始化完成, timecost: 0 ms
[2025-08-08 20:32:19.154] [mirror] [info] [ChineseG2p.cc:28] ChineseG2P 开始初始化...
[2025-08-08 20:32:19.155] [mirror] [info] [ChineseG2p.cc:49] ChineseG2P 初始化成功, timecost: 0 ms
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
Can't open file:.cachefile
Load Cache file error.
[2025-08-08 20:32:19.851] [mirror] [info] [ChineseBert.cc:53] bert模型加载成功: ../data/tts/./common/mnn_models/chinese_bert.mnn
Can't open file:.cachefile
Load Cache file error.
[2025-08-08 20:32:20.383] [mirror] [info] [EnglishBert.cc:39] en_bert模型加载成功: ../data/tts/./common/mnn_models/english_bert.mnn
Can't open file:.tts_generator_cachefile
Load Cache file error.
[2025-08-08 20:32:20.811] [mirror] [info] [TTSGenerator.cc:42] tts 模型加载成功: ../data/tts/tts_generator_w_bert_chenxi_0310_int8.mnn
[2025-08-08 20:32:20.811] [mirror] [info] [TTSGenerator.cc:43] ### tts load memory increase : 44.42533
[2025-08-08 20:32:26.467] [mirror] [info] [TTSGenerator.cc:99] ### tts forward memory increase : 61.616356.

```
  - (b) 测试Sherpa-MNN-ASR模块：先下载模型，对应地址：[sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20](https://modelscope.cn/models/MNN/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20)，一样的，我这里将其解压之后，放到了项目底下的data/asr路径，具体看代码
```bash
>> ./TestMnnASRSession \
    --tokens=../data/asr/tokens.txt \
    --encoder=../data/asr/encoder.mnn \
    --decoder=../data/asr/decoder.mnn \
    --joiner=../data/asr/joiner.mnn \
    --num-threads=2 \
    ../data/asr/1.wav
[2025-08-08 20:41:08.084] [mirror] [info] [ParseOptions.cc:310] ./TestMnnASRSession --tokens=../data/asr/tokens.txt --encoder=../data/asr/encoder.mnn --decoder=../data/asr/decoder.mnn --joiner=../data/asr/joiner.mnn --num-threads=2 ../data/asr/1.wav 

OnlineMnnASRSessionConfig(feature_extractor_config=FeatureExtractorConfig(sampling_rate=16000, feature_dim=80, low_freq=20, high_freq=-400, dither=0, normalize_samples=True, snip_edges=False), model_config=ModelConfig(encoder="../data/asr/encoder.mnn", decoder="../data/asr/decoder.mnn", joiner="../data/asr/joiner.mnn"), endpoint_config=EndpointConfig(rule1=EndpointRule(must_contain_nonsilence=False, min_trailing_silence=2.4, min_utterance_length=0), rule2=EndpointRule(must_contain_nonsilence=True, min_trailing_silence=1.2, min_utterance_length=0), rule3=EndpointRule(must_contain_nonsilence=False, min_trailing_silence=0, min_utterance_length=20)), enable_endpoint=True, blank_penalty=0, temperature_scale=2")
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
[2025-08-08 20:41:10.449] [mirror] [info] [MnnASRSession.cc:115] processed result: Elapsed seconds: 0.36, Audio duration (s): 5.5, Real time factor (RTF) = 0.36/5.5 = 0.065
欢迎大家来体验达摩院推出的语音识别模型
{ "text": "欢迎大家来体验达摩院推出的语音识别模型", "tokens": ["欢", "迎", "大", "家", "来", "体", "验", "达", "摩", "院", "推", "出", "的", "语", "音", "识", "别", "模", "型"], "timestamps": [0.96, 1.20, 1.60, 1.80, 2.12, 2.36, 2.60, 2.88, 3.00, 3.20, 3.40, 3.68, 3.88, 4.16, 4.36, 4.56, 4.84, 5.12, 5.40], "segment": 0, "words": [], "start_time": 0.00, "is_final": false}


processed succeed.
{ "text": "欢迎大家来体验达摩院推出的语音识别模型", "tokens": ["欢", "迎", "大", "家", "来", "体", "验", "达", "摩", "院", "推", "出", "的", "语", "音", "识", "别", "模", "型"], "timestamps": [0.96, 1.20, 1.60, 1.80, 2.12, 2.36, 2.60, 2.88, 3.00, 3.20, 3.40, 3.68, 3.88, 4.16, 4.36, 4.56, 4.84, 5.12, 5.40], "segment": 0, "words": [], "start_time": 0.00, "is_final": false}
```
  - (c) 测试MNN-LLM模块：先下载模型，对应地址：[Qwen2.5-1.5B-Instruct-MNN](https://modelscope.cn/models/MNN/Qwen2.5-1.5B-Instruct-MNN)，我这里将其解压之后，放到了项目底下的data/llm路径，具体看代码
```bash
>> ./TestMnnLLMSession  
[2025-08-08 20:43:29.898] [mirror] [info] [TestMnnLLMSession.cc:38] cfg: {"is_r1":false,"max_new_tokens":2048,"minP":0.05000000074505806,"mixed_samplers":["topK","topP","minP","temperature"],"penalty":1.2,"precision":"high","sampler_type":"mixed","system_prompt":"You are a helpful assistant.","temperature":0.6000000238418579,"topK":20,"topP":0.949999988079071}.
[2025-08-08 20:43:29.898] [mirror] [info] [TestMnnLLMSession.cc:44] extra cfg: {"mmap_dir":"./tmp","use_mmap":false}
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
[2025-08-08 20:43:29.899] [mirror] [info] [MnnLLMSession.cc:178] extra_cfg: {"is_r1":false,"max_new_tokens":2048,"minP":0.05000000074505806,"mixed_samplers":["topK","topP","minP","temperature"],"penalty":1.2,"precision":"high","sampler_type":"mixed","system_prompt":"You are a helpful assistant.","temperature":0.6000000238418579,"tmp_path":"./tmp","topK":20,"topP":0.949999988079071,"use_mmap":true}

User: 你好
[2025-08-08 20:43:37.192] [mirror] [info] [TestMnnLLMSession.cc:14] response: 你好
[2025-08-08 20:43:37.246] [mirror] [info] [TestMnnLLMSession.cc:14] response: ！
[2025-08-08 20:43:37.284] [mirror] [info] [TestMnnLLMSession.cc:14] response: 很高兴
[2025-08-08 20:43:37.322] [mirror] [info] [TestMnnLLMSession.cc:14] response: 能
[2025-08-08 20:43:37.358] [mirror] [info] [TestMnnLLMSession.cc:14] response: 为你
[2025-08-08 20:43:37.395] [mirror] [info] [TestMnnLLMSession.cc:14] response: 服务
[2025-08-08 20:43:37.429] [mirror] [info] [TestMnnLLMSession.cc:14] response: 。

Assistant: 你好！很高兴能为你服务。

User: 你是什么大模型
[2025-08-08 20:43:44.603] [mirror] [info] [TestMnnLLMSession.cc:14] response: 我是
[2025-08-08 20:43:44.636] [mirror] [info] [TestMnnLLMSession.cc:14] response: 来自
[2025-08-08 20:43:44.672] [mirror] [info] [TestMnnLLMSession.cc:14] response: 阿里
[2025-08-08 20:43:44.706] [mirror] [info] [TestMnnLLMSession.cc:14] response: 云
[2025-08-08 20:43:44.740] [mirror] [info] [TestMnnLLMSession.cc:14] response: 的
[2025-08-08 20:43:44.774] [mirror] [info] [TestMnnLLMSession.cc:14] response: 通
[2025-08-08 20:43:44.810] [mirror] [info] [TestMnnLLMSession.cc:14] response: 义
[2025-08-08 20:43:44.849] [mirror] [info] [TestMnnLLMSession.cc:14] response: 千
[2025-08-08 20:43:44.887] [mirror] [info] [TestMnnLLMSession.cc:14] response: 问
[2025-08-08 20:43:44.919] [mirror] [info] [TestMnnLLMSession.cc:14] response: ，
[2025-08-08 20:43:44.953] [mirror] [info] [TestMnnLLMSession.cc:14] response: 是一个
[2025-08-08 20:43:44.990] [mirror] [info] [TestMnnLLMSession.cc:14] response: 预
[2025-08-08 20:43:45.023] [mirror] [info] [TestMnnLLMSession.cc:14] response: 训练
[2025-08-08 20:43:45.058] [mirror] [info] [TestMnnLLMSession.cc:14] response: 语言
[2025-08-08 20:43:45.099] [mirror] [info] [TestMnnLLMSession.cc:14] response: 模型
[2025-08-08 20:43:45.134] [mirror] [info] [TestMnnLLMSession.cc:14] response: ，
[2025-08-08 20:43:45.170] [mirror] [info] [TestMnnLLMSession.cc:14] response: 可以帮助
[2025-08-08 20:43:45.209] [mirror] [info] [TestMnnLLMSession.cc:14] response: 提供
[2025-08-08 20:43:45.248] [mirror] [info] [TestMnnLLMSession.cc:14] response: 文本
[2025-08-08 20:43:45.285] [mirror] [info] [TestMnnLLMSession.cc:14] response: 生成
[2025-08-08 20:43:45.318] [mirror] [info] [TestMnnLLMSession.cc:14] response: 、
[2025-08-08 20:43:45.352] [mirror] [info] [TestMnnLLMSession.cc:14] response: 翻译
[2025-08-08 20:43:45.387] [mirror] [info] [TestMnnLLMSession.cc:14] response: 、
[2025-08-08 20:43:45.419] [mirror] [info] [TestMnnLLMSession.cc:14] response: 摘要
[2025-08-08 20:43:45.453] [mirror] [info] [TestMnnLLMSession.cc:14] response: 、
[2025-08-08 20:43:45.485] [mirror] [info] [TestMnnLLMSession.cc:14] response: 问答
[2025-08-08 20:43:45.521] [mirror] [info] [TestMnnLLMSession.cc:14] response: 等
[2025-08-08 20:43:45.556] [mirror] [info] [TestMnnLLMSession.cc:14] response: 服务
[2025-08-08 20:43:45.602] [mirror] [info] [TestMnnLLMSession.cc:14] response: 。

Assistant: 我是来自阿里云的通义千问，是一个预训练语言模型，可以帮助提供文本生成、翻译、摘要、问答等服务。
```
- (d) 测试MNN-A2BS模块：先下载模型，对应地址：[UniTalker-MNN](https://modelscope.cn/models/MNN/UniTalker-MNN)，我这里将其解压之后，放到了项目底下的data/a2bs路径，具体看代码
```bash
>> ./TestMnnA2BSSession
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
[2025-08-11 11:15:56.179] [mirror] [info] [AudioToFrameBlendShape.cc:72] ### audio_to_flame_blend_shape load memory increase : 364.81076
[2025-08-11 11:15:56.179] [mirror] [info] [AudioTo3dgsBlendShape.cc:27] A2BSService ParseInputsFromJson execution time: 0 ms
Load a2bs recources successed.
Audio format: 2, Channels: 1, Sample rate: 44100
[2025-08-11 11:15:56.585] [mirror] [info] [AudioToFrameBlendShape.cc:108] ### audio2verts forward memory increase : 35.944977
[2025-08-11 11:15:56.585] [mirror] [info] [AudioTo3dgsBlendShape.cc:70] Audio2BS timecost: 405.000000 ms, audio_duration: 2799.977295 ms, rtf:(0.144644+0.144644)
```

&emsp;注意，这里A2BS不知道结果对不对，等后面NNR模块开源才能知晓。

## 3.下一步计划
* A2BS模块结果是否正确需要移植完[NNR模块](https://github.com/alibaba/MNN/issues/361)才能验证，但是NNR模块没有开源，所以就这样吧，懒得整了。

## 4.参考资料
* [1] [MNN](https://github.com/alibaba/MNN)