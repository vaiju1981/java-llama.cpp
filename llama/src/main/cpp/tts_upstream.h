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

// Forward declarations only. This shared interface header names nlohmann::ordered_json once (the
// get_tts_version() speaker parameter) but never instantiates it, so it must not pull the full
// ~25k-line <nlohmann/json.hpp> into every translation unit that includes it. The single caller that
// constructs the empty-object default (tts_engine.cpp) includes the full <nlohmann/json.hpp> itself.
#include <nlohmann/json_fwd.hpp>

#include "common.h" // llama_tokens
#include "llama.h"  // llama_model, llama_vocab, llama_token

// Mirrors the upstream enum (identical definition; ODR-compatible across translation units). The
// generated TU carries upstream's own copy, so these enumerators and their order MUST stay
// token-identical to upstream — otherwise the two definitions assign different integer values to the
// same name (a silent miscompile). cmake/generate-tts-upstream.cmake asserts the upstream enum still
// reads `{ OUTETTS_V0_2, OUTETTS_V0_3 }` at configure time and fails loud (pointing here) if a
// llama.cpp bump changes it.
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

// No default argument here on purpose: constructing nlohmann::ordered_json::object() needs the full
// json definition, which this header deliberately does not include (see the json_fwd note above). The
// sole caller (tts_engine.cpp) passes an explicit empty object; the generated TU keeps upstream's own
// default, so its internal calls are unaffected.
outetts_version get_tts_version(llama_model *model, nlohmann::ordered_json speaker);

// Default OuteTTS speaker profile, extracted from upstream main() into the generated TU.
extern const std::string jllama_tts_default_audio_text;
extern const std::string jllama_tts_default_audio_data;

#endif // JLLAMA_TTS_UPSTREAM_H
