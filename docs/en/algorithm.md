# Algorithm

`openvad` is a frame-based adaptive VAD. The design goal is a fast, dependency
light baseline that behaves predictably in batch and low-latency settings.

## Pipeline

1. Convert input audio to mono `float32` in `[-1, 1]`.
2. Split into overlapping frames, defaulting to 20 ms frames and 10 ms hops.
3. Compute per-frame RMS energy, energy in dBFS, zero-crossing rate, and crest
   factor.
4. Estimate the initial noise floor from the lower energy percentile.
5. Convert relative energy above the adaptive noise floor into a speech
   probability.
6. Penalize frames that look like impulse noise, near-DC signal, or high-rate
   noise.
7. Apply onset/offset hysteresis.
8. Fill short gaps, remove short islands, and pad final speech regions.

## Why This Shape

Energy-only VAD is fast but tends to chatter at speech boundaries and fails when
the background level drifts. `openvad` keeps the speed of energy VAD while adding
three stabilizers:

- Adaptive noise tracking updates primarily outside confident speech.
- Hysteresis uses separate enter and exit thresholds.
- Segment post-processing enforces minimum useful speech and silence durations.

## Complexity

Runtime is linear in the number of samples. Memory usage is linear in the number
of frames, not the number of samples. The C++ core performs the sample-level loop
and returns compact NumPy arrays to Python.

## Practical Limits

The detector is not a substitute for a trained neural VAD on difficult acoustic
conditions. Expect weaker performance on:

- Speech under loud music.
- Rapidly changing industrial noise.
- Far-field microphones with very low signal-to-noise ratio.
- Non-speech sounds with speech-like energy envelopes.

Use the frame arrays in `VadResult.frames` to inspect failures and tune
thresholds for your domain.
