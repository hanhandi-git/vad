# Tuning Guide

Start with the default configuration, inspect false positives and false
negatives, then adjust one parameter at a time.

## Common Adjustments

| Symptom | Adjustment |
| --- | --- |
| Speech starts are clipped | Increase `speech_pad_ms` or lower `onset_threshold`. |
| Speech tails are clipped | Lower `offset_threshold` or increase `speech_pad_ms`. |
| Background noise becomes speech | Increase `aggressiveness` or `onset_threshold`. |
| Short commands are missed | Lower `min_speech_ms`. |
| Speech is split into many segments | Increase `min_silence_ms`. |
| Non-speech clicks are detected | Increase `min_speech_ms` or `aggressiveness`. |

## Suggested Profiles

Permissive:

```python
VoiceActivityDetector(aggressiveness=0, onset_threshold=0.52, offset_threshold=0.36)
```

Balanced:

```python
VoiceActivityDetector(aggressiveness=1)
```

Strict:

```python
VoiceActivityDetector(aggressiveness=2, onset_threshold=0.64, offset_threshold=0.48)
```

## Evaluation

For serious deployments, create labeled intervals for representative audio and
measure frame-level precision/recall after converting labels to the same hop
size as the detector. Segment-level metrics are also useful when the downstream
task only cares about whether speech chunks are usable.
