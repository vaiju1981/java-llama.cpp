// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Interface to the OuteTTS helpers that the JNI engine links against. The definitions are NOT
// here and NOT hand-copied: cmake/generate-tts-upstream.cmake derives them at build time from the
// pinned llama.cpp tools/tts/tts.cpp (which declares them `static`), giving them external linkage.
// This header only declares the call surface — interface facts, not the upstream implementation —
// so tts_engine.cpp can call them. If upstream changes a signature, the generator's de-static
// assert fails the configure; if it changes a type, this header stops matching and the link fails.

#ifndef JLLAMA_TTS_UPSTREAM_H
#define JLLAMA_TTS_UPSTREAM_H

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "common.h" // llama_tokens
#include "llama.h"  // llama_model, llama_vocab, llama_token

// Mirrors the upstream enum (identical definition; ODR-compatible across translation units).
enum outetts_version { OUTETTS_V0_2, OUTETTS_V0_3 };

// --- derived from upstream tts.cpp (defined in the generated translation unit) ---

// Spectral synthesis: codec embeddings -> float PCM. The number-to-words / prompt formatting that
// the upstream CLI applies is preserved (process_text calls replace_numbers_with_words), so digits
// are spoken rather than dropped.
std::vector<float> embd_to_audio(const float *embd, const int n_codes, const int n_embd, const int n_thread);

std::string process_text(const std::string &text, outetts_version tts_version);

void prompt_add(llama_tokens &prompt, const llama_tokens &tokens);
void prompt_add(llama_tokens &prompt, const llama_vocab *vocab, const std::string &txt, bool add_special,
                bool parse_special);
void prompt_init(llama_tokens &prompt, const llama_vocab *vocab);

std::vector<llama_token> prepare_guide_tokens(const llama_vocab *vocab, const std::string &str,
                                              outetts_version tts_version);

outetts_version get_tts_version(llama_model *model, nlohmann::ordered_json speaker = nlohmann::ordered_json::object());

// Default OuteTTS speaker profile, extracted from upstream main() into the generated TU.
extern const std::string jllama_tts_default_audio_text;
extern const std::string jllama_tts_default_audio_data;

#endif // JLLAMA_TTS_UPSTREAM_H
