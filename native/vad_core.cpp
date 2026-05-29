#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

struct FrameFeature {
    float energy_db;
    float probability;
    float zcr;
};

float clamp(float value, float lo, float hi) {
    return std::max(lo, std::min(value, hi));
}

float sigmoid(float value) {
    if (value >= 0.0f) {
        const float z = std::exp(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(value);
    return z / (1.0f + z);
}

float percentile(std::vector<float> values, float p) {
    if (values.empty()) {
        return -120.0f;
    }
    p = clamp(p, 0.0f, 1.0f);
    const size_t idx = static_cast<size_t>(std::round(p * static_cast<float>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + idx, values.end());
    return values[idx];
}

std::vector<uint8_t> post_process(
    const std::vector<uint8_t>& raw,
    int min_speech_frames,
    int min_silence_frames,
    int pad_frames
) {
    const int n = static_cast<int>(raw.size());
    std::vector<uint8_t> out = raw;

    int i = 0;
    while (i < n) {
        while (i < n && out[i] == 1) {
            ++i;
        }
        const int gap_start = i;
        while (i < n && out[i] == 0) {
            ++i;
        }
        const int gap_end = i;
        if (gap_start > 0 && gap_end < n && (gap_end - gap_start) <= min_silence_frames) {
            std::fill(out.begin() + gap_start, out.begin() + gap_end, 1);
        }
    }

    i = 0;
    while (i < n) {
        while (i < n && out[i] == 0) {
            ++i;
        }
        const int run_start = i;
        while (i < n && out[i] == 1) {
            ++i;
        }
        const int run_end = i;
        if (run_end > run_start && (run_end - run_start) < min_speech_frames) {
            std::fill(out.begin() + run_start, out.begin() + run_end, 0);
        }
    }

    if (pad_frames <= 0) {
        return out;
    }

    std::vector<uint8_t> padded = out;
    i = 0;
    while (i < n) {
        while (i < n && out[i] == 0) {
            ++i;
        }
        const int run_start = i;
        while (i < n && out[i] == 1) {
            ++i;
        }
        const int run_end = i;
        if (run_end > run_start) {
            const int start = std::max(0, run_start - pad_frames);
            const int end = std::min(n, run_end + pad_frames);
            std::fill(padded.begin() + start, padded.begin() + end, 1);
        }
    }
    return padded;
}

py::dict analyze(
    py::array_t<float, py::array::c_style | py::array::forcecast> samples,
    int sample_rate,
    float frame_ms,
    float hop_ms,
    float onset_threshold,
    float offset_threshold,
    int min_speech_ms,
    int min_silence_ms,
    int speech_pad_ms,
    int aggressiveness
) {
    if (sample_rate <= 0) {
        throw std::invalid_argument("sample_rate must be positive");
    }
    if (frame_ms <= 0.0f || hop_ms <= 0.0f) {
        throw std::invalid_argument("frame_ms and hop_ms must be positive");
    }
    if (offset_threshold > onset_threshold) {
        throw std::invalid_argument("offset_threshold must be <= onset_threshold");
    }
    aggressiveness = static_cast<int>(clamp(static_cast<float>(aggressiveness), 0.0f, 3.0f));

    const auto view = samples.unchecked<1>();
    const ssize_t sample_count = view.shape(0);
    const int frame_len = std::max(1, static_cast<int>(std::round(sample_rate * frame_ms / 1000.0f)));
    const int hop_len = std::max(1, static_cast<int>(std::round(sample_rate * hop_ms / 1000.0f)));
    if (sample_count <= 0 || sample_count < frame_len) {
        py::dict empty;
        empty["speech"] = py::array_t<uint8_t>({0});
        empty["probability"] = py::array_t<float>({0});
        empty["energy_db"] = py::array_t<float>({0});
        empty["zcr"] = py::array_t<float>({0});
        empty["frame_samples"] = frame_len;
        empty["hop_samples"] = hop_len;
        return empty;
    }

    const int frame_count = 1 + static_cast<int>((sample_count - frame_len) / hop_len);
    std::vector<FrameFeature> features(frame_count);
    std::vector<float> energies;
    energies.reserve(frame_count);

    for (int frame = 0; frame < frame_count; ++frame) {
        const ssize_t start = static_cast<ssize_t>(frame) * hop_len;
        double sum_sq = 0.0;
        float peak = 0.0f;
        int crossings = 0;
        float prev = view(start);

        for (int j = 0; j < frame_len; ++j) {
            const float value = view(start + j);
            sum_sq += static_cast<double>(value) * static_cast<double>(value);
            peak = std::max(peak, std::abs(value));
            if (j > 0 && ((value >= 0.0f) != (prev >= 0.0f))) {
                ++crossings;
            }
            prev = value;
        }

        const float mean_sq = static_cast<float>(sum_sq / static_cast<double>(frame_len));
        const float rms = std::sqrt(mean_sq);
        const float energy_db = 20.0f * std::log10(rms + 1.0e-12f);
        const float zcr = static_cast<float>(crossings) / static_cast<float>(std::max(1, frame_len - 1));
        const float crest_db = 20.0f * std::log10((peak + 1.0e-12f) / (rms + 1.0e-12f));

        features[frame] = {energy_db, 0.0f, zcr};
        energies.push_back(energy_db);

        // Store crest temporarily in probability; it is overwritten after the global noise estimate.
        features[frame].probability = crest_db;
    }

    const float noise_floor = percentile(energies, 0.20f);
    const float high_floor = percentile(energies, 0.90f);
    const float dynamic_range = std::max(1.0f, high_floor - noise_floor);
    const float noise_margin = 5.0f + 1.5f * static_cast<float>(aggressiveness);
    const float slope = 2.4f + 0.35f * static_cast<float>(aggressiveness);

    py::array_t<float> probability({frame_count});
    py::array_t<float> energy_db({frame_count});
    py::array_t<float> zcr({frame_count});
    py::array_t<uint8_t> speech({frame_count});

    auto probability_view = probability.mutable_unchecked<1>();
    auto energy_view = energy_db.mutable_unchecked<1>();
    auto zcr_view = zcr.mutable_unchecked<1>();
    auto speech_view = speech.mutable_unchecked<1>();

    std::vector<uint8_t> raw(frame_count, 0);
    bool active = false;
    float adaptive_noise = noise_floor;

    for (int i = 0; i < frame_count; ++i) {
        const float e = features[i].energy_db;
        const float z = features[i].zcr;
        const float crest_db = features[i].probability;
        const float relative = (e - adaptive_noise - noise_margin) / slope;
        float p = sigmoid(relative);

        if (dynamic_range < 4.0f) {
            p *= 0.45f;
        }
        if (z < 0.003f) {
            p *= 0.72f;
        } else if (z > 0.30f) {
            p *= 0.62f;
        }
        if (crest_db > 24.0f) {
            p *= 0.70f;
        }
        if (e < -72.0f) {
            p = 0.0f;
        }
        p = clamp(p, 0.0f, 1.0f);

        const float threshold = active ? offset_threshold : onset_threshold;
        active = p >= threshold;
        raw[i] = active ? 1 : 0;

        if (!active || p < 0.35f) {
            adaptive_noise = 0.995f * adaptive_noise + 0.005f * e;
            adaptive_noise = std::min(adaptive_noise, e + 8.0f);
        }

        probability_view(i) = p;
        energy_view(i) = e;
        zcr_view(i) = z;
    }

    const int min_speech_frames = std::max(1, static_cast<int>(std::ceil(min_speech_ms / hop_ms)));
    const int min_silence_frames = std::max(1, static_cast<int>(std::ceil(min_silence_ms / hop_ms)));
    const int pad_frames = std::max(0, static_cast<int>(std::round(speech_pad_ms / hop_ms)));
    const std::vector<uint8_t> processed =
        post_process(raw, min_speech_frames, min_silence_frames, pad_frames);

    for (int i = 0; i < frame_count; ++i) {
        speech_view(i) = processed[i];
    }

    py::dict result;
    result["speech"] = speech;
    result["probability"] = probability;
    result["energy_db"] = energy_db;
    result["zcr"] = zcr;
    result["frame_samples"] = frame_len;
    result["hop_samples"] = hop_len;
    result["noise_floor_db"] = noise_floor;
    return result;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "High-performance C++ core for openvad.";
    m.def(
        "analyze",
        &analyze,
        py::arg("samples"),
        py::arg("sample_rate"),
        py::arg("frame_ms") = 20.0f,
        py::arg("hop_ms") = 10.0f,
        py::arg("onset_threshold") = 0.58f,
        py::arg("offset_threshold") = 0.42f,
        py::arg("min_speech_ms") = 80,
        py::arg("min_silence_ms") = 120,
        py::arg("speech_pad_ms") = 40,
        py::arg("aggressiveness") = 1
    );
}
