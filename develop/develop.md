当前没有“按语言种类”自动配置参数。openvad 现在是声学统计 VAD，主要看能量、噪声底、ZCR、迟滞和后处理，不识别语言，也不会因为英语/日
  语/中文自动换参数。实际更应该按“场景”调参：干净近讲、远场会议、电话、背景音乐、噪声、短指令、长对话等，而不是只按语言。

  当前可调参数在 VadConfig：

  ┌──────────────────┬────────┬─────────────────────────────────────────────┐
  │ 参数             │ 默认值 │ 作用                                        │
  ├──────────────────┼────────┼─────────────────────────────────────────────┤
  │ frame_ms         │   20.0 │ 分析帧长，越短延迟越低但更不稳。            │
  │ hop_ms           │   10.0 │ 帧移，影响时间分辨率和计算量。              │
  │ onset_threshold  │   0.58 │ 进入语音的概率阈值，越高越保守。            │
  │ offset_threshold │   0.42 │ 维持语音的概率阈值，越低越不容易切断尾音。  │
  │ min_speech_ms    │     80 │ 删除过短语音段，抑制点击声/噪声。           │
  │ min_silence_ms   │    120 │ 合并语音中的短静音，减少碎片化。            │
  │ speech_pad_ms    │     40 │ 语音段前后扩展，避免切掉字头字尾。          │
  │ aggressiveness   │      1 │ 0 最宽松，3 最严格；内部会提高噪声 margin。 │
  └──────────────────┴────────┴─────────────────────────────────────────────┘

  建议不要先做“语言 preset”，而是做“domain preset”：

  - near_field_clean：默认或 aggressiveness=1
  - noisy：aggressiveness=2，提高 onset_threshold
  - short_command：降低 min_speech_ms，增加 speech_pad_ms
  - meeting_far_field：增加 min_silence_ms 和 speech_pad_ms
  - music_background：当前统计 VAD 会比较吃力，建议后续加 neural/ONNX backend

  业界 benchmark 可以这样选：

  ┌────────────────────────┬───────────────────────────────────────────────────┬───────────────────────────────────────────────────┐
  │ Benchmark              │ 适合验证什么                                      │ 备注                                              │
  ├────────────────────────┼───────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
  │ AVA-Speech             │ 影视/YouTube 场景，干净语音、语音+噪声、语音+音乐 │ Google 公开数据，约 45h、约 46K 标注段，常用于    │
  │                        │                                                   │ VAD/SAD 论文。来源：https://research.google.com/  │
  │                        │                                                   │ ava/index.html                                    │
  │ DIHARD                 │ 多领域困难 SAD/diarization                        │ 很适合压力测试，但部分数据通过 LDC 获取。来源：   │
  │                        │                                                   │ https://dihardchallenge.github.io/dihard1/        │
  │                        │                                                   │ data.html                                         │
  │ NIST OpenSAT / OpenSAD │ 正式 SAD 评测范式                                 │ 适合看官方评测规则和指标。来源：https://          │
  │                        │                                                   │ www.nist.gov/programs-projects/speech-analytics   │
  │ AMI Meeting Corpus     │ 会议、远场、多说话人                              │ 约 100h 会议数据，适合会议 VAD。来源：https://    │
  │                        │                                                   │ groups.inf.ed.ac.uk/ami/corpus                    │
  │ VoxConverse            │ YouTube 多说话人、重叠语音、真实噪声              │ 约 50h，偏 diarization，但可从 RTTM 转 SAD。来    │
  │                        │                                                   │ 源：https://mmai.io/datasets/voxconverse/         │
  │ MUSAN                  │ 噪声/音乐/语音增强与合成测试                      │ 更适合作为噪声和音乐混合源，不是边界级 VAD        │
  │                        │                                                   │ benchmark。来源：https://arxiv.org/abs/1510.08484 │
  │ AISHELL-4              │ 中文会议场景                                      │ 普通话会议，多说话人，适合中文会议类验证。来源：  │
  │                        │                                                   │ https://arxiv.org/abs/2104.03603                  │
  └────────────────────────┴───────────────────────────────────────────────────┴───────────────────────────────────────────────────┘

  指标建议至少跑这些：

  - Frame-level：precision、recall、F1，按 10ms 或 20ms 对齐。
  - Detection Error Rate：(false alarm + missed detection) / reference speech duration，pyannote.metrics 也把它作为 SAD 主要指标之
    一：https://pyannote.github.io/pyannote-metrics/reference.html
  - False alarm seconds / Miss seconds：比单个 F1 更容易定位问题。
  - ROC/AUC：扫 onset_threshold / offset_threshold 看可调空间。
  - Boundary error：起点/终点偏差，建议加 100-250ms collar 单独看。
  - Runtime：real-time factor、CPU 占用、流式延迟。

  你当前项目里已经有基础验证工具，可以直接接 benchmark manifest：

  uv run python tools/evaluate_dataset.py data/manifest.jsonl
  uv run python tools/sweep_thresholds.py data/manifest.jsonl
  uv run python tools/inspect_file.py sample.wav --output report.html

  下一步我建议加一个 VadPreset 层，先按场景而不是语言内置几组参数；然后把 AVA-Speech 或 AMI/VoxConverse 的标注转换成当前
  manifest.jsonl 格式，跑 sweep_thresholds.py 找一组真实数据上的默认参数。