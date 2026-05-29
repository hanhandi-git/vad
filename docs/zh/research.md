# VAD 业界方案调研

最后检查日期：2026-05-29。

VAD，即 Voice Activity Detection，也常被称为 SAD（Speech Activity
Detection），目标是判断音频帧或音频区间里是否有人类语音。现代系统里，
VAD 通常不只是一个独立功能，而是用于控制 ASR 端点检测、语音智能体轮次、
说话人分离、降噪、录音压缩或唤醒链路。

## 主流思路

### 1. 能量与规则型 VAD

这是最早、也仍然常用的一类方法。它通常组合短时能量、RMS、过零率、谱通量、
SNR 估计、基频线索、hangover 逻辑和自适应噪声跟踪。

典型流程：

- 每 10-30 ms 计算一帧特征。
- 从低能量帧估计噪声底。
- 当能量或 SNR 高于自适应阈值时判为语音。
- 使用迟滞阈值和 hangover 避免边界抖动。
- 合并短静音，删除短语音岛。

优点：

- 很快，适合 C/C++ 实现。
- 不需要模型文件或训练数据。
- 延迟和内存占用可预测。
- 在干净语音或稳定噪声下是很好的 baseline。

缺点：

- 容易受音乐、脉冲噪声和非平稳噪声影响。
- 难以区分语音和“像语音”的其他声音。
- 精度强依赖阈值和目标场景。

当前 `openvad` 就属于这一类：它使用自适应噪声跟踪、迟滞阈值和段落后处理。

### 2. 经典统计型 VAD

很多传统生产级 VAD 会使用滤波器组特征和统计分类器，例如 GMM，并配合手写
状态机做时间平滑。

代表方案：

- WebRTC VAD 支持 8/16/32 kHz 下 10/20/30 ms 的帧级判决，并提供四档
  aggressiveness。WebRTC 源码里包含 `vad_core.c`、`vad_filterbank.c`
  和 `vad_gmm.c`，体现了紧凑的定点滤波器组/GMM 实现路线。

优点：

- 非常成熟且高效。
- 适合实时通信和嵌入式场景。
- 不需要 GPU 或大型推理运行时。

缺点：

- 通常输出二值帧结果，概率校准能力弱。
- 在困难噪声下通常不如现代神经网络 VAD。
- 帧长和采样率约束可能带来工程限制。

### 3. 无监督鲁棒 VAD

无监督方法试图在没有监督训练数据的情况下提升鲁棒性。rVAD 是代表方案之一：
它使用多轮降噪、SNR 加权能量差、基频检查和段级判决。

优点：

- 在部分噪声场景下比简单能量阈值更稳。
- 不需要标注训练集。
- 适合语料清洗和研究流水线。

缺点：

- 比能量法复杂。
- 不一定适合严格低延迟流式场景。
- 在广泛真实场景下通常仍弱于训练良好的神经网络模型。

### 4. 神经网络帧分类器

现代 VAD 经常使用神经网络处理 log-mel、filter-bank 或原始波形特征。常见
结构包括 DNN/MLP、CNN、RNN/LSTM/GRU、TCN 和 transformer 风格编码器。
输出通常是每帧或每个 chunk 的语音概率。

代表方案：

- Google 的 RNN VAD 工作表明，联合优化时间连续性和声学分类后，RNN 可以
  超过更大的 GMM 与手写平滑状态机 baseline。
- Silero VAD 是常用的预训练神经网络 VAD，提供 PyTorch、ONNX、浏览器、
  C++、Rust、Go、Java 等生态集成。
- Picovoice Cobra 是商业化端侧神经网络 VAD SDK，强调本地执行和跨平台部署。

优点：

- 通常更适合真实噪声和多样麦克风。
- 概率输出便于阈值调节。
- 可以导出到 ONNX、CoreML、TFLite 或自定义 native runtime。

缺点：

- 需要模型文件和推理运行时选择。
- 需要有代表性的训练和评估数据。
- 量化、流式状态和跨平台部署要谨慎处理。
- 离开训练域后性能可能明显波动。

### 5. 面向说话人分离的语音分割

在 diarization 系统里，VAD 往往是更大分割模型的一部分，同时处理说话人变化、
重叠语音等任务。pyannote.audio 是典型例子：它提供可训练的神经网络组件和
预训练模型，覆盖说话人分离、语音活动检测、说话人变化检测、重叠语音检测和
说话人嵌入。

优点：

- 当下游任务是说话人分离而不只是 ASR gating 时更合适。
- 可以在语音活动、说话人变化、重叠检测和 embedding 之间共享特征/模型。
- pyannote.audio 这类工具支持在目标领域上 fine-tune。

缺点：

- 比简单 VAD 重很多。
- 训练或高吞吐推理通常更偏 GPU。
- 边界可能更适合 diarization 指标，而不一定适合实时语音智能体轮次。

### 6. ASR 端点检测与 blank-token 逻辑

流式 ASR 系统常用模型输出而不是单独声学 VAD 做端点检测。NVIDIA Riva 文档中
描述了基于 ASR 非 blank/blank 帧窗口的 utterance 起止检测，同时也支持在 ASR
pipeline 中加入神经网络 VAD。

典型流程：

- 最近窗口里有足够比例的非 blank 帧时，认为 utterance 开始。
- 最近窗口里大部分帧为 blank 或静音时，认为 utterance 结束。
- 可选地在 ASR 前或 ASR 内加入神经网络 VAD。

优点：

- 与 ASR decoder 的行为对齐。
- 能减少“声学上有声音但 ASR 没有有效 token”的误触发。
- 适合生产 ASR endpointing。

缺点：

- 绑定特定 ASR 模型和解码栈。
- 如果目标是在 ASR 前节省算力，单靠它不够。
- 因为要等待模型证据，端点可能更晚。

### 7. 面向语音智能体的语义轮次检测

语音智能体越来越需要区分“声学 VAD”和“用户是否说完”。OpenAI Realtime 提供
`server_vad` 与 `semantic_vad`：前者基于静音切分音频，后者由模型判断用户
是否完成了一句话。这能处理用户中途停顿、犹豫词或拖尾等情况。

优点：

- 比单纯静音检测更适合对话轮次。
- 能减少 speech-to-speech 场景里过早打断用户。
- 可以通过 eagerness 调整响应速度和耐心程度。

缺点：

- 依赖模型侧推理和产品集成。
- 不能直接替代本地声学 VAD。
- 不适合完全本地、强隐私的流水线。

## 当前工程上的常见组合

生产系统里，VAD 往往是分层的：

- 用便宜的本地能量/WebRTC 类 gate 过滤明显静音。
- 用神经网络 VAD 输出更可靠的语音概率。
- 用 ASR endpointing 或语义轮次检测判断一句话是否真正结束。
- 用 prefix padding、最短语音、最短静音和 hangover 修正边界。

这种分层设计很常见，因为“低算力”“强鲁棒”“边界准确”“对话自然”其实是
不同目标，很难由一个 VAD 模块同时最优解决。

## 对 `openvad` 的启发

建议方向：

- 保留当前统计核心，作为高速 baseline。
- 持续暴露概率曲线和帧级特征，方便评估与调参。
- 如果能接受采样率/帧长约束，可以增加 WebRTC 兼容模式。
- 增加可选 ONNX 神经网络后端，用于更高噪声场景。
- 将 endpointing 策略与帧级分类器分开。
- 修改默认参数前，先完善评估工具和真实数据集验证。

应避免：

- 把 VAD 精度当成一个全场景通用指标。
- 只优化合成测试。
- 没有目标场景评估就硬编码阈值。
- 在同一个 API 里混合声学语音检测和语义说完判断。

## 资料来源

- WebRTC VAD 源码目录：<https://webrtc.googlesource.com/src/+/main/common_audio/vad>
- WebRTC VAD 接口：<https://webrtc.googlesource.com/src/webrtc/+/f54860e9ef0b68e182a01edc994626d21961bc4b/common_audio/vad/include/vad.h>
- WebRTC audio-processing VAD：<https://webrtc.googlesource.com/src/+/bab128555afa0f94994a5d5689b7d8da930cdee1/modules/audio_processing/vad/voice_activity_detector.h>
- Silero VAD：<https://github.com/snakers4/silero-vad>
- pyannote.audio：<https://github.com/pyannote/pyannote-audio>
- pyannote.audio 论文：<https://arxiv.org/abs/1911.01255>
- rVAD 论文：<https://arxiv.org/abs/1906.03588>
- Google RNN VAD：<https://research.google/pubs/recurrent-neural-networks-for-voice-activity-detection/>
- OpenAI Realtime VAD：<https://platform.openai.com/docs/guides/realtime-vad>
- NVIDIA Riva ASR endpointing 与 neural VAD：<https://docs.nvidia.com/deeplearning/riva/archives/2-15-1/public/asr/asr-pipeline-configuration.html>
- Picovoice Cobra VAD：<https://picovoice.ai/docs/cobra/>
