// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 The llama.cpp authors
//
// SPDX-License-Identifier: MIT
//
// Text-to-speech output DSP, vendored from llama.cpp tools/tts/tts.cpp (the standalone
// `llama-tts` CLI, which is not built as a library). These functions are pure signal
// processing — no llama / ggml / JNI state — so they live in a header-only helper that the
// JNI bridge and the C++ unit tests can both include. Kept byte-faithful to upstream so a
// llama.cpp version bump is a mechanical re-sync; only `save_wav16` (file write) is replaced
// by `pcm_to_wav16_bytes` (in-memory), since the JNI layer returns WAV bytes to Java.

#ifndef JLLAMA_TTS_DSP_HPP
#define JLLAMA_TTS_DSP_HPP

#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace jllama_tts {

// --- vendored verbatim from tts.cpp (fill_hann_window/twiddle/irfft/fold/embd_to_audio) ---

inline void fill_hann_window(int length, bool periodic, float *output) {
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}

// very poor-man fft
inline void twiddle(float *real, float *imag, int k, int N) {
    float angle = 2 * M_PI * k / N;
    *real = cos(angle);
    *imag = sin(angle);
}

inline void irfft(int n, const float *inp_cplx, float *out_real) {
    int N = n / 2 + 1;

    std::vector<float> real_input(N);
    std::vector<float> imag_input(N);
    for (int i = 0; i < N; ++i) {
        real_input[i] = inp_cplx[2 * i];
        imag_input[i] = inp_cplx[2 * i + 1];
    }

    std::vector<float> real_output(n);
    std::vector<float> imag_output(n);

    for (int k = 0; k < n; ++k) {
        real_output[k] = 0.0f;
        imag_output[k] = 0.0f;
        for (int m = 0; m < N; ++m) {
            float twiddle_real;
            float twiddle_imag;

            twiddle(&twiddle_real, &twiddle_imag, k * m, n);

            real_output[k] += real_input[m] * twiddle_real - imag_input[m] * twiddle_imag;
            imag_output[k] += real_input[m] * twiddle_imag + imag_input[m] * twiddle_real;
        }
    }

    for (int i = 0; i < n; ++i) {
        out_real[i] = real_output[i] / N;
    }
}

inline void fold(const std::vector<float> &data, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad,
                 std::vector<float> &output) {
    int64_t output_height = n_out;
    int64_t kernel_w = n_win;
    int64_t stride_w = n_hop;
    int64_t width = n_out;

    output.resize(width, 0.0f);

    int64_t col_idx = 0;
    for (int64_t w_col = 0; w_col < width; ++w_col) {
        int64_t start = w_col * stride_w - n_pad;
        int64_t end = start + kernel_w;

        for (int64_t w_im = start; w_im < end; ++w_im) {
            if (w_im >= 0 && w_im < output_height && col_idx < (int64_t)data.size()) {
                output[w_im] += data[col_idx];
            }
            col_idx++;
        }
    }

    output.resize(n_out - 2 * n_pad);
}

// TODO: not optimized at all
inline std::vector<float> embd_to_audio(const float *embd, const int n_codes, const int n_embd, const int n_thread) {
    const int n_fft = 1280;
    const int n_hop = 320;
    const int n_win = 1280;
    const int n_pad = (n_win - n_hop) / 2;
    const int n_out = (n_codes - 1) * n_hop + n_win;

    std::vector<float> hann(n_fft);

    fill_hann_window(hann.size(), true, hann.data());

    int n_spec = n_embd * n_codes;

    std::vector<float> E(n_spec);
    std::vector<float> S(n_spec);
    std::vector<float> ST(n_spec);

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd; ++k) {
            E[k * n_codes + l] = embd[l * n_embd + k];
        }
    }

    for (int k = 0; k < n_embd / 2; ++k) {
        for (int l = 0; l < n_codes; ++l) {
            float mag = E[(k)*n_codes + l];
            float phi = E[(k + n_embd / 2) * n_codes + l];

            mag = exp(mag);

            if (mag > 1e2) {
                mag = 1e2;
            }
            S[2 * (k * n_codes + l) + 0] = mag * cosf(phi);
            S[2 * (k * n_codes + l) + 1] = mag * sinf(phi);
        }
    }

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd / 2; ++k) {
            ST[l * n_embd + 2 * k + 0] = S[2 * (k * n_codes + l) + 0];
            ST[l * n_embd + 2 * k + 1] = S[2 * (k * n_codes + l) + 1];
        }
    }

    std::vector<float> res(n_codes * n_fft);
    std::vector<float> hann2(n_codes * n_fft);

    const int threads = n_thread > 0 ? n_thread : 1;
    std::vector<std::thread> workers(threads);
    for (int i = 0; i < threads; ++i) {
        workers[i] = std::thread([&, i]() {
            for (int l = i; l < n_codes; l += threads) {
                irfft(n_fft, ST.data() + l * n_embd, res.data() + l * n_fft);
                for (int j = 0; j < n_fft; ++j) {
                    res[l * n_fft + j] *= hann[j];
                    hann2[l * n_fft + j] = hann[j] * hann[j];
                }
            }
        });
    }
    for (int i = 0; i < threads; ++i) {
        workers[i].join();
    }

    std::vector<float> audio;
    std::vector<float> env;

    fold(res, n_out, n_win, n_hop, n_pad, audio);
    fold(hann2, n_out, n_win, n_hop, n_pad, env); // TODO: can be done once

    for (size_t i = 0; i < audio.size(); ++i) {
        audio[i] /= env[i];
    }

    return audio;
}

// --- in-memory replacement for tts.cpp's file-writing save_wav16 ---

// Encode float PCM samples (range ~[-1, 1]) as a 16-bit mono WAV byte stream. Mirrors the
// header layout and clamping of tts.cpp's save_wav16, but returns bytes instead of writing a file.
inline std::vector<uint8_t> pcm_to_wav16_bytes(const std::vector<float> &data, int sample_rate) {
    const uint16_t num_channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t data_size = (uint32_t)(data.size() * (bits_per_sample / 8));
    const uint32_t byte_rate = (uint32_t)sample_rate * num_channels * (bits_per_sample / 8);
    const uint16_t block_align = num_channels * (bits_per_sample / 8);
    const uint32_t chunk_size = 36 + data_size;

    std::vector<uint8_t> out;
    out.reserve(44 + data_size);

    auto put_u32 = [&out](uint32_t v) {
        out.push_back((uint8_t)(v & 0xff));
        out.push_back((uint8_t)((v >> 8) & 0xff));
        out.push_back((uint8_t)((v >> 16) & 0xff));
        out.push_back((uint8_t)((v >> 24) & 0xff));
    };
    auto put_u16 = [&out](uint16_t v) {
        out.push_back((uint8_t)(v & 0xff));
        out.push_back((uint8_t)((v >> 8) & 0xff));
    };
    auto put_tag = [&out](const char *tag) {
        out.push_back((uint8_t)tag[0]);
        out.push_back((uint8_t)tag[1]);
        out.push_back((uint8_t)tag[2]);
        out.push_back((uint8_t)tag[3]);
    };

    put_tag("RIFF");
    put_u32(chunk_size);
    put_tag("WAVE");
    put_tag("fmt ");
    put_u32(16); // PCM fmt-chunk size
    put_u16(1);  // audio format = PCM
    put_u16(num_channels);
    put_u32((uint32_t)sample_rate);
    put_u32(byte_rate);
    put_u16(block_align);
    put_u16(bits_per_sample);
    put_tag("data");
    put_u32(data_size);

    for (const float sample : data) {
        double scaled = (double)sample * 32767.0;
        scaled = scaled < -32768.0 ? -32768.0 : (scaled > 32767.0 ? 32767.0 : scaled);
        int16_t pcm = (int16_t)scaled;
        out.push_back((uint8_t)(pcm & 0xff));
        out.push_back((uint8_t)((pcm >> 8) & 0xff));
    }

    return out;
}

} // namespace jllama_tts

#endif // JLLAMA_TTS_DSP_HPP
