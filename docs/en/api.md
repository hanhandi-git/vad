# API

## `VoiceActivityDetector`

```python
from openvad import VoiceActivityDetector

detector = VoiceActivityDetector(aggressiveness=1)
result = detector.analyze(samples, sample_rate=16_000)
```

`samples` must be a mono one-dimensional array-like object. Values are expected
to be normalized floating-point PCM samples, usually in `[-1, 1]`.

## `detect`

Convenience wrapper:

```python
from openvad import detect

result = detect(samples, 16_000)
```

## `detect_file`

Reads PCM WAV and runs VAD:

```python
from openvad import detect_file

result = detect_file("speech.wav")
```

Supported WAV encodings are unsigned 8-bit PCM and signed 16/24/32-bit PCM.
Multi-channel files are downmixed to mono by averaging channels.

## Result Types

`VadResult` contains:

- `sample_rate`: input sample rate.
- `samples`: input sample count.
- `duration`: input duration in seconds.
- `segments`: list of speech `Segment` objects.
- `frames`: frame-level `FrameAnalysis`.

`Segment` contains:

- `start`: seconds.
- `end`: seconds.
- `duration`: seconds.
- `confidence`: mean frame probability in the segment.

`FrameAnalysis` contains NumPy arrays:

- `speech`: final binary frame labels.
- `probability`: raw speech probabilities.
- `energy_db`: frame energy in dBFS.
- `zcr`: zero-crossing rate.
