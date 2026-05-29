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

## `StreamingVoiceActivityDetector`

Use the streaming detector when PCM arrives in chunks:

```python
from openvad import StreamingVoiceActivityDetector

def on_start(event):
    print("start", event.time)


def on_end(event):
    print("end", event.time, event.segment)


stream = StreamingVoiceActivityDetector(
    sample_rate=16_000,
    on_start_of_speech=on_start,
    on_end_of_speech=on_end,
    aggressiveness=1,
)

for chunk in pcm_chunks:
    stream.push_pcm(chunk, sample_format="s16le")

stream.flush()
```

`push(samples)` accepts normalized mono float PCM arrays. `push_pcm(bytes)`
accepts raw `s16le`, `s16be`, `f32le`, or `u8` PCM and downmixes multi-channel
input when `channels` is greater than `1`. `push` methods return only stable
completed segments; call `flush()` once at end of stream to emit remaining
segments. Use `push_events()`, `push_pcm_events()`, and `flush_events()` when you
prefer returned `start_of_speech` / `end_of_speech` events instead of callbacks.

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
