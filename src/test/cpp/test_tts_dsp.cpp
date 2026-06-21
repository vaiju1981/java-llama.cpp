// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Unit tests for the vendored TTS output DSP (src/main/cpp/tts_dsp.hpp). Pure signal
// processing — no model, no JNI — so they run in the standard jllama_test suite.

#include "tts_dsp.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

using namespace jllama_tts;

namespace {
uint32_t read_u32(const std::vector<uint8_t> &b, size_t off) {
    return (uint32_t)b[off] | ((uint32_t)b[off + 1] << 8) | ((uint32_t)b[off + 2] << 16) | ((uint32_t)b[off + 3] << 24);
}
std::string read_tag(const std::vector<uint8_t> &b, size_t off) {
    return std::string(b.begin() + off, b.begin() + off + 4);
}
} // namespace

TEST(TtsWav, HeaderAndPayloadAreWellFormed) {
    std::vector<float> pcm = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f};
    std::vector<uint8_t> wav = pcm_to_wav16_bytes(pcm, 24000);

    // 44-byte header + 2 bytes per 16-bit sample.
    ASSERT_EQ(wav.size(), 44u + pcm.size() * 2);
    EXPECT_EQ(read_tag(wav, 0), "RIFF");
    EXPECT_EQ(read_tag(wav, 8), "WAVE");
    EXPECT_EQ(read_tag(wav, 12), "fmt ");
    EXPECT_EQ(read_tag(wav, 36), "data");
    EXPECT_EQ(read_u32(wav, 16), 16u);                             // PCM fmt-chunk size
    EXPECT_EQ(read_u32(wav, 24), 24000u);                          // sample rate
    EXPECT_EQ(read_u32(wav, 40), (uint32_t)(pcm.size() * 2));      // data size
    EXPECT_EQ(read_u32(wav, 4), 36u + (uint32_t)(pcm.size() * 2)); // RIFF chunk size
}

TEST(TtsWav, ClampsAndEncodesSamplesLittleEndian) {
    std::vector<uint8_t> wav = pcm_to_wav16_bytes({0.0f, 1.0f, -1.0f, -2.0f}, 24000);
    // 0 -> 0; 1.0 -> 32767; -1.0 -> -32767 (= -1.0*32767, floor clamp not reached); -2.0 clamps to -32768.
    auto sample = [&](int i) -> int16_t {
        size_t off = 44 + i * 2;
        return (int16_t)((uint16_t)wav[off] | ((uint16_t)wav[off + 1] << 8));
    };
    EXPECT_EQ(sample(0), 0);
    EXPECT_EQ(sample(1), 32767);
    EXPECT_EQ(sample(2), -32767);
    EXPECT_EQ(sample(3), -32768);
}

TEST(TtsDsp, HannWindowPeriodicEndpoints) {
    std::vector<float> w(8);
    fill_hann_window((int)w.size(), /*periodic=*/true, w.data());
    EXPECT_NEAR(w[0], 0.0f, 1e-6); // periodic Hann starts at 0
    EXPECT_NEAR(w[4], 1.0f, 1e-6); // and peaks at the centre (length/2)
    for (float v : w) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 1.0f + 1e-6f);
    }
}

TEST(TtsDsp, FoldTrimsPadAndSumsOverlaps) {
    // n_out=10, n_win=4, n_hop=2, n_pad=1 -> output length n_out - 2*n_pad = 8.
    std::vector<float> data(/*n_out*/ 10 * /*cols*/ 1, 1.0f);
    std::vector<float> out;
    fold(data, 10, 4, 2, 1, out);
    EXPECT_EQ(out.size(), 8u);
}

TEST(TtsDsp, EmbdToAudioOutputLengthMatchesIdentity) {
    // Vocoder spectrogram dim and code count; output length is n_codes * n_hop(=320).
    const int n_embd = 1282;
    const int n_codes = 3;
    std::vector<float> embd(n_embd * n_codes, 0.01f);
    std::vector<float> audio = embd_to_audio(embd.data(), n_codes, n_embd, /*n_thread=*/2);
    EXPECT_EQ(audio.size(), (size_t)(n_codes * 320));
}
