# 调参指南

建议从默认配置开始，先观察误检和漏检，再一次只调整一个参数。

## 常见问题

| 现象 | 调整建议 |
| --- | --- |
| 语音开头被截断 | 增大 `speech_pad_ms` 或降低 `onset_threshold`。 |
| 语音结尾被截断 | 降低 `offset_threshold` 或增大 `speech_pad_ms`。 |
| 背景噪声被判成语音 | 增大 `aggressiveness` 或 `onset_threshold`。 |
| 短指令漏检 | 降低 `min_speech_ms`。 |
| 一句话被切成很多段 | 增大 `min_silence_ms`。 |
| 点击声等非语音被检出 | 增大 `min_speech_ms` 或 `aggressiveness`。 |

## 推荐配置

宽松：

```python
VoiceActivityDetector(aggressiveness=0, onset_threshold=0.52, offset_threshold=0.36)
```

均衡：

```python
VoiceActivityDetector(aggressiveness=1)
```

严格：

```python
VoiceActivityDetector(aggressiveness=2, onset_threshold=0.64, offset_threshold=0.48)
```

## 评估方式

严肃上线前，应准备目标场景的标注音频，将标注区间转换到与检测器相同的帧移，
再评估帧级 precision、recall 和 F1。如果下游任务只关心语音片段是否可用，也
应补充段级指标。
