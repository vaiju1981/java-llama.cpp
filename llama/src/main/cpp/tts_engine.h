// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Native text-to-speech engine: a self-contained orchestration of the two-model OuteTTS pipeline
// (TTC OuteTTS LLM -> audio codec tokens; CTS WavTokenizer vocoder -> embeddings -> embd_to_audio ->
// 16-bit WAV), single-stream (n_parallel = 1). Kept out of jllama.cpp so the JNI layer stays thin.

#ifndef JLLAMA_TTS_ENGINE_H
#define JLLAMA_TTS_ENGINE_H

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace jllama_tts {

// Opaque handle owning both loaded models / contexts. Created by engine_init, freed by engine_free.
struct tts_engine;

// Load the TTC (OuteTTS text-to-codes) and CTS (codes-to-speech vocoder) models. Returns nullptr and
// sets `err` on failure.
tts_engine *engine_init(const std::string &ttc_model_path, const std::string &cts_model_path, int n_gpu_layers,
                        int n_threads, std::string &err);

// Synthesize `text` to a 24 kHz mono 16-bit WAV byte stream in `out_wav`. Returns false and sets `err`
// on failure. Thread-compatible but not re-entrant on the same engine instance.
bool engine_synthesize(tts_engine *engine, const std::string &text, int n_predict, int top_k, uint32_t seed,
                       std::vector<uint8_t> &out_wav, std::string &err);

// Release both models / contexts. Safe on nullptr.
void engine_free(tts_engine *engine);

// ---------------------------------------------------------------------------
// OuteTTS audio-codec token filtering (pure transform, testable without a model).
//
// The TTC model emits codec tokens in a high id window; the CTS vocoder only
// understands the 0-based codec ids inside [k_outetts_codec_lo, k_outetts_codec_hi].
// This window is identical for OuteTTS V0_2 and V0_3 (verified against upstream
// tools/tts/tts.cpp), so it is NOT gated on tts_version. Keep only tokens in the
// window and rebase them to 0-based ids.
constexpr int32_t k_outetts_codec_lo = 151672;
constexpr int32_t k_outetts_codec_hi = 155772;

inline void filter_outetts_codec_tokens(std::vector<int32_t> &tokens) {
    tokens.erase(std::remove_if(tokens.begin(), tokens.end(),
                                [](int32_t t) { return t < k_outetts_codec_lo || t > k_outetts_codec_hi; }),
                 tokens.end());
    for (auto &t : tokens) {
        t -= k_outetts_codec_lo;
    }
}

} // namespace jllama_tts

#endif // JLLAMA_TTS_ENGINE_H
