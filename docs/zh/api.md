# API

## `VoiceActivityDetector`

```python
from openvad import VoiceActivityDetector

detector = VoiceActivityDetector(aggressiveness=1)
result = detector.analyze(samples, sample_rate=16_000)
```

`samples` 必须是单声道一维数组。推荐传入归一化浮点 PCM，通常范围为
`[-1, 1]`。

## `detect`

便捷函数：

```python
from openvad import detect

result = detect(samples, 16_000)
```

## `detect_file`

读取 PCM WAV 并执行 VAD：

```python
from openvad import detect_file

result = detect_file("speech.wav")
```

支持 unsigned 8-bit PCM 与 signed 16/24/32-bit PCM。多声道文件会通过通道
平均下混为单声道。

## 结果类型

`VadResult` 包含：

- `sample_rate`：输入采样率。
- `samples`：输入样本数。
- `duration`：输入时长，单位秒。
- `segments`：语音段 `Segment` 列表。
- `frames`：帧级 `FrameAnalysis`。

`Segment` 包含：

- `start`：开始时间，单位秒。
- `end`：结束时间，单位秒。
- `duration`：持续时间，单位秒。
- `confidence`：该段平均帧概率。

`FrameAnalysis` 包含 NumPy 数组：

- `speech`：后处理后的二值帧标签。
- `probability`：原始语音概率。
- `energy_db`：帧能量，单位 dBFS。
- `zcr`：过零率。
