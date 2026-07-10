// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Unit tests for the in-memory WAV writer (src/main/cpp/tts_wav.hpp) — our own code, not upstream.
// The OuteTTS DSP it pairs with (embd_to_audio etc.) is derived from upstream tts.cpp at build time
// and exercised end-to-end by the Java TtsIntegrationTest, not unit-tested here.

#include "tts_wav.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include "tts_engine.h"

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

// ============================================================
// filter_ouetts_codec_tokens (M3 — V0_2 / V0_3 codec window)
// ============================================================

// The OuteTTS codec window is identical for V0_2 and V0_3 (verified against upstream
// tools/tts/tts.cpp). In-range tokens are kept and rebased to 0-based codec ids.
TEST(OuteTtsCodecFilter, InRangeTokensAreRebased) {
    std::vector<int32_t> codes = {151672, 151673, 155772};
    filter_ouetts_codec_tokens(codes);
    ASSERT_EQ(codes.size(), 3u);
    EXPECT_EQ(codes[0], 0);
    EXPECT_EQ(codes[1], 1);
    EXPECT_EQ(codes[2], 155772 - 151672);
}

// Tokens below / above the window (e.g. text, control, or speaker tokens) are dropped.
TEST(OuteTtsCodecFilter, OutOfRangeTokensAreDropped) {
    std::vector<int32_t> codes = {151671, 198 /*newline*/, 155773, 151672};
    filter_ouetts_codec_tokens(codes);
    ASSERT_EQ(codes.size(), 1u);
    EXPECT_EQ(codes[0], 0);
}

TEST(OuteTtsCodecFilter, EmptyInput_StaysEmpty) {
    std::vector<int32_t> codes;
    filter_ouetts_codec_tokens(codes);
    EXPECT_TRUE(codes.empty());
}

TEST(OuteTtsCodecFilter, V02AndV03ShareWindow) {
    // Whatever the engine version, the same window applies (M3). Exercise both boundaries.
    std::vector<int32_t> v02 = {151672, 153000, 155772};
    std::vector<int32_t> v03 = {151672, 153000, 155772};
    filter_ouetts_codec_tokens(v02);
    filter_ouetts_codec_tokens(v03);
    EXPECT_EQ(v02, v03);
    EXPECT_EQ(v02, (std::vector<int32_t>{0, 153000 - 151672, 155772 - 151672}));
}
