# VAD Industry Survey

Last reviewed: 2026-05-29.

Voice Activity Detection (VAD), also called Speech Activity Detection (SAD), is
used to decide whether an audio frame or region contains human speech. In modern
systems it is rarely a standalone feature only; it usually controls ASR
endpointing, voice-agent turn taking, speaker diarization, noise suppression,
recording compression, or wake-up pipelines.

## Main Families

### 1. Energy and Rule-Based VAD

This is the oldest and still useful family. It combines short-time energy, RMS,
zero-crossing rate, spectral flux, SNR estimates, pitch cues, hangover logic,
and adaptive noise tracking.

Typical design:

- Compute frame features every 10-30 ms.
- Estimate the noise floor from low-energy frames.
- Detect speech when energy or SNR rises above an adaptive threshold.
- Use hysteresis and hangover to avoid boundary chatter.
- Merge short gaps and remove short islands.

Strengths:

- Very fast and easy to implement in C/C++.
- No model files or training data.
- Predictable latency and memory usage.
- Good baseline for clean speech or stable noise.

Weaknesses:

- Sensitive to music, impulsive noise, and non-stationary noise.
- Cannot reliably distinguish speech from other speech-like sounds.
- Accuracy depends heavily on thresholds and target domain.

`openvad` currently belongs to this family, with adaptive noise tracking and
post-processing.

### 2. Classical Statistical VAD

Classical production VADs often use filter-bank features and statistical
classifiers such as Gaussian Mixture Models (GMMs), sometimes with hand-tuned
state machines for temporal smoothing.

Representative example:

- WebRTC VAD exposes 10/20/30 ms frame decisions at 8/16/32 kHz and four
  aggressiveness levels. The WebRTC source tree includes `vad_core.c`,
  `vad_filterbank.c`, and `vad_gmm.c`, showing a compact fixed-point,
  filter-bank/GMM style implementation.

Strengths:

- Extremely mature and efficient.
- Suitable for real-time communications and embedded use.
- No GPU or large runtime dependency.

Weaknesses:

- Binary frame output gives less calibration flexibility than probability
  scores.
- Often less robust than modern neural VAD in difficult noise.
- Frame-size and sample-rate constraints can be awkward.

### 3. Unsupervised Robust VAD

Unsupervised methods try to improve robustness without supervised neural
training. rVAD is a representative example: it uses denoising passes, SNR
weighted energy differences, pitch checks, and segment-level decisions.

Strengths:

- More robust than simple energy thresholds in some noisy conditions.
- No labeled training set required.
- Good fit for corpus preparation and research pipelines.

Weaknesses:

- More complex than energy VAD.
- Not always ideal for strict low-latency streaming.
- Usually still weaker than well-trained neural models on broad real-world
  audio.

### 4. Neural Frame Classifiers

Modern VADs frequently use neural networks over log-mel, filter-bank, or raw
waveform features. Common architectures include DNN/MLP, CNN, RNN/LSTM/GRU,
TCN, and transformer-style encoders. They output a speech probability per frame
or per chunk.

Representative examples:

- Google showed RNN VAD can outperform a larger GMM plus hand-tuned smoothing
  baseline by optimizing temporal continuity and acoustic classification
  jointly.
- Silero VAD is a widely used pretrained neural VAD with PyTorch, ONNX,
  browser, C++, Rust, Go, Java, and other community integrations.
- Picovoice Cobra is a commercial on-device neural VAD SDK that emphasizes local
  execution and cross-platform deployment.

Strengths:

- Usually better under real-world noise and diverse microphones.
- Probability output is useful for threshold tuning.
- Can be exported to ONNX/CoreML/TFLite/custom native runtimes.

Weaknesses:

- Requires model files and runtime decisions.
- Needs representative training/evaluation data.
- Quantization and streaming state must be handled carefully.
- Performance can vary sharply outside the training domain.

### 5. Diarization-Oriented Speech Segmentation

In diarization systems, VAD is often part of a broader segmentation model that
also detects speaker changes or overlapped speech. pyannote.audio is a common
example: it provides trainable neural building blocks and pretrained models for
speaker diarization, speech activity detection, speaker change detection,
overlapped speech detection, and speaker embeddings.

Strengths:

- Better fit when the downstream task is diarization, not just ASR gating.
- Can share features/models across speech activity, speaker change, overlap,
  and embeddings.
- Fine-tuning on target domains is supported by toolkits like pyannote.audio.

Weaknesses:

- Heavier than simple VAD.
- Often GPU-oriented for training or high-throughput inference.
- Segment boundaries may be optimized for diarization metrics rather than
  real-time voice-agent turn taking.

### 6. ASR Endpointing and Blank-Token Logic

Streaming ASR systems often implement endpointing using model outputs rather
than a separate acoustic VAD. NVIDIA Riva documents utterance start/end
detection based on windows of nonblank/blank ASR frames, and also supports a
neural VAD component in the ASR pipeline.

Typical design:

- Start an utterance when enough recent ASR frames are nonblank.
- Stop an utterance when most recent frames are blank or silence-like.
- Optionally add a neural VAD before or alongside the ASR model.

Strengths:

- Closely aligned with the ASR decoder.
- Reduces cases where acoustic speech is present but ASR has no useful tokens.
- Useful for production ASR endpointing.

Weaknesses:

- Tied to a particular ASR model and decoding stack.
- Less useful if VAD is needed before ASR for compute reduction.
- Endpointing may lag because it waits for model evidence.

### 7. Semantic Turn Detection for Voice Agents

Voice agents increasingly distinguish acoustic VAD from end-of-turn detection.
OpenAI Realtime exposes `server_vad`, which chunks audio based on silence, and
`semantic_vad`, which uses model judgment about whether the user has completed
an utterance. This addresses cases where the user pauses mid-thought or trails
off with filler words.

Strengths:

- Better conversation turn-taking than silence-only VAD.
- Reduces premature interruption in speech-to-speech agents.
- Can tune eagerness for faster or more patient response timing.

Weaknesses:

- Requires model-side inference and product integration.
- Not a drop-in replacement for local acoustic VAD.
- Less suitable for privacy-sensitive fully local pipelines.

## Current Practical Pattern

In production, VAD is often layered:

- A cheap local energy/WebRTC-style gate removes obvious silence.
- A neural VAD produces calibrated speech probabilities.
- ASR endpointing or semantic turn detection decides when an utterance is
  complete.
- Post-processing adds prefix padding, minimum speech duration, minimum silence,
  and hangover.

This layered design is common because no single VAD objective covers all needs:
low compute, robust detection, accurate boundaries, and natural conversation
turn-taking are different targets.

## Design Implications for `openvad`

Good next steps:

- Keep the current statistical core as the fast baseline.
- Expose probability traces and frame features for evaluation and tuning.
- Add optional WebRTC-compatible mode only if sample-rate/frame-size constraints
  are acceptable.
- Add an optional ONNX neural backend for higher-noise use cases.
- Keep endpointing policy separate from frame classification.
- Add evaluation tooling before changing defaults.

Avoid:

- Treating VAD accuracy as one universal metric.
- Optimizing only synthetic tests.
- Hard-coding thresholds without domain-specific evaluation.
- Mixing acoustic speech detection and semantic turn completion in one API.

## Source Notes

- WebRTC VAD source tree: <https://webrtc.googlesource.com/src/+/main/common_audio/vad>
- WebRTC VAD interface: <https://webrtc.googlesource.com/src/webrtc/+/f54860e9ef0b68e182a01edc994626d21961bc4b/common_audio/vad/include/vad.h>
- WebRTC audio-processing VAD: <https://webrtc.googlesource.com/src/+/bab128555afa0f94994a5d5689b7d8da930cdee1/modules/audio_processing/vad/voice_activity_detector.h>
- Silero VAD: <https://github.com/snakers4/silero-vad>
- pyannote.audio: <https://github.com/pyannote/pyannote-audio>
- pyannote.audio paper: <https://arxiv.org/abs/1911.01255>
- rVAD paper: <https://arxiv.org/abs/1906.03588>
- Google RNN VAD: <https://research.google/pubs/recurrent-neural-networks-for-voice-activity-detection/>
- OpenAI Realtime VAD: <https://platform.openai.com/docs/guides/realtime-vad>
- NVIDIA Riva ASR endpointing and neural VAD: <https://docs.nvidia.com/deeplearning/riva/archives/2-15-1/public/asr/asr-pipeline-configuration.html>
- Picovoice Cobra VAD: <https://picovoice.ai/docs/cobra/>
