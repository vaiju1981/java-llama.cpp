# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# Build-time extractor for the OuteTTS native pipeline.
#
# Rather than hand-copying functions out of llama.cpp's tools/tts/tts.cpp (a maintenance
# burden that silently diverges on every upgrade), this script DERIVES a compilable
# translation unit MECHANICALLY from the pinned upstream source at configure time. The
# generated file is never committed (it lives in the build tree) and is regenerated from
# whatever tts.cpp the pinned GIT_TAG resolves to, so divergence is impossible and an
# upstream bump is picked up automatically.
#
# What it does:
#   1. Keeps everything in tts.cpp BEFORE `int main(` (includes, the outetts_version enum,
#      the number-to-words tables, and the DSP + prompt + text helpers). main() itself —
#      the standalone CLI entry point — is dropped (that is why tts.cpp cannot simply be
#      added to target_sources: its main() would clash at link time).
#   2. Strips `static` from exactly the helpers the JNI engine calls, giving them external
#      linkage so tts_engine.cpp can link against them (they are `static`/internal upstream).
#   3. Extracts the two hard-coded default-speaker literals (audio_text / audio_data), which
#      upstream embeds as locals inside main(), into two external constants.
#
# Every anchor is asserted: if upstream renames a function or moves the literals, the
# configure step FAILS LOUDLY with a pointer here, the same fail-on-drift contract as
# patches/. Inputs (via -D): TTS_SRC, OUT_CPP, LLAMA_TAG.

if(NOT EXISTS "${TTS_SRC}")
    message(FATAL_ERROR "generate-tts-upstream: upstream tts.cpp not found at '${TTS_SRC}'")
endif()

file(READ "${TTS_SRC}" SRC)

# --- 1. keep the pre-main() portion (main() is the unbuildable-as-library CLI entry point) ---
string(FIND "${SRC}" "\nint main(" MAIN_POS)
if(MAIN_POS EQUAL -1)
    message(FATAL_ERROR "generate-tts-upstream: 'int main(' anchor not found in tts.cpp — upstream layout changed; update cmake/generate-tts-upstream.cmake")
endif()
string(SUBSTRING "${SRC}" 0 ${MAIN_POS} PREMAIN)

# --- 2. give external linkage to the helpers the JNI engine calls ---
# Each entry is asserted present as `static <sig>` before stripping, so an upstream rename
# fails the configure instead of silently dropping the symbol (caught later only at link).
set(JLLAMA_TTS_DESTATIC
    "std::vector<float> embd_to_audio("
    "std::string process_text("
    "void prompt_add("
    "void prompt_init("
    "std::vector<llama_token> prepare_guide_tokens("
    "outetts_version get_tts_version(")
foreach(sig IN LISTS JLLAMA_TTS_DESTATIC)
    string(FIND "${PREMAIN}" "static ${sig}" _pos)
    if(_pos EQUAL -1)
        message(FATAL_ERROR "generate-tts-upstream: expected 'static ${sig}' in upstream tts.cpp but it is absent — upstream changed; update the de-static list in cmake/generate-tts-upstream.cmake")
    endif()
    string(REPLACE "static ${sig}" "${sig}" PREMAIN "${PREMAIN}")
endforeach()

# --- 2a. pin the outetts_version enum against the hand-written copy in tts_upstream.h ---
# src/main/cpp/tts_upstream.h re-declares `enum outetts_version { OUTETTS_V0_2, OUTETTS_V0_3 }` because
# it cannot include the generated TU. The two definitions live in different translation units and must
# stay token-identical: if upstream reorders/renames/extends the enum, the generated TU and the header
# would bind the same name to different integer values (a silent miscompile). Capture the upstream enum
# body and compare its enumerator list so a drift fails the configure with a pointer to update the header.
string(REGEX MATCH "enum[ \t\r\n]+outetts_version[ \t\r\n]*{([^}]*)}" _enum_match "${PREMAIN}")
if(_enum_match STREQUAL "")
    message(FATAL_ERROR "generate-tts-upstream: 'enum outetts_version' not found in tts.cpp — upstream changed; update cmake/generate-tts-upstream.cmake and src/main/cpp/tts_upstream.h")
endif()
set(_enum_body "${CMAKE_MATCH_1}")
string(REGEX REPLACE "//[^\n]*" "" _enum_body "${_enum_body}")  # strip any line comments
string(REGEX REPLACE "[ \t\r\n]+" "" _enum_body "${_enum_body}") # strip all whitespace
string(REGEX REPLACE ",+$" "" _enum_body "${_enum_body}")       # strip a trailing comma
if(NOT _enum_body STREQUAL "OUTETTS_V0_2,OUTETTS_V0_3")
    message(FATAL_ERROR "generate-tts-upstream: upstream 'enum outetts_version' enumerators are now '${_enum_body}' (expected 'OUTETTS_V0_2,OUTETTS_V0_3'). Update the matching enum in src/main/cpp/tts_upstream.h to keep the two definitions ODR-identical, then update this assertion in cmake/generate-tts-upstream.cmake")
endif()

# --- 2b. verify BOTH prompt_add overloads that tts_upstream.h declares are present ---
# `void prompt_add(` is shared by three upstream overloads; the de-static REPLACE above (correctly) gives
# all of them external linkage, but the single string(FIND) only proves >=1 exists. tts_upstream.h
# declares exactly two — (llama_tokens&, const llama_tokens&) and the (vocab, txt, add_special,
# parse_special) builder — and tts_engine.cpp links against them. Pin both here (whitespace-tolerant) so
# dropping or renaming either fails the configure with a clear pointer instead of a cryptic link error.
if(NOT PREMAIN MATCHES "void[ \t]+prompt_add[ \t]*\\([^)]*const[ \t]+llama_tokens[ \t]*&[ \t]*tokens[ \t]*\\)")
    message(FATAL_ERROR "generate-tts-upstream: the prompt_add(llama_tokens&, const llama_tokens&) overload declared in src/main/cpp/tts_upstream.h was not found in tts.cpp — upstream changed; update the de-static list and src/main/cpp/tts_upstream.h")
endif()
if(NOT PREMAIN MATCHES "void[ \t]+prompt_add[ \t]*\\([^)]*vocab[^)]*add_special[^)]*parse_special[^)]*\\)")
    message(FATAL_ERROR "generate-tts-upstream: the prompt_add(llama_tokens&, const llama_vocab*, const std::string&, bool, bool) overload declared in src/main/cpp/tts_upstream.h was not found in tts.cpp — upstream changed; update the de-static list and src/main/cpp/tts_upstream.h")
endif()

# --- 3. extract the two default-speaker literals from inside main() ---
# audio_text: a single-line  std::string audio_text = "<|text_start|>the<|text_sep|>...";
# The leading "<|text_start|>the<|text_sep|>" disambiguates it from the empty-seed literal
# in audio_text_from_speaker(). Content runs to the next double-quote (it embeds none).
set(_AT_DECL "std::string audio_text = \"")
string(FIND "${SRC}" "${_AT_DECL}<|text_start|>the<|text_sep|>" _at_at)
if(_at_at EQUAL -1)
    message(FATAL_ERROR "generate-tts-upstream: default audio_text literal not found in tts.cpp main() — upstream changed; update cmake/generate-tts-upstream.cmake")
endif()
string(LENGTH "${_AT_DECL}" _at_decl_len)
math(EXPR _at_content "${_at_at} + ${_at_decl_len}")
string(SUBSTRING "${SRC}" ${_at_content} -1 _at_rest)
string(FIND "${_at_rest}" "\"" _at_len)
string(SUBSTRING "${_at_rest}" 0 ${_at_len} AUDIO_TEXT)

# audio_data: a multi-line raw string  std::string audio_data = R"(...)";
# The R"( form disambiguates it from the empty-seed "..." literal in audio_data_from_speaker().
# Content runs to the first )" (the body embeds none — only <|...|> tokens).
set(_AD_DECL "std::string audio_data = R\"(")
string(FIND "${SRC}" "${_AD_DECL}" _ad_at)
if(_ad_at EQUAL -1)
    message(FATAL_ERROR "generate-tts-upstream: default audio_data raw-string literal not found in tts.cpp main() — upstream changed; update cmake/generate-tts-upstream.cmake")
endif()
string(LENGTH "${_AD_DECL}" _ad_decl_len)
math(EXPR _ad_content "${_ad_at} + ${_ad_decl_len}")
string(SUBSTRING "${SRC}" ${_ad_content} -1 _ad_rest)
string(FIND "${_ad_rest}" ")\"" _ad_len)
string(SUBSTRING "${_ad_rest}" 0 ${_ad_len} AUDIO_DATA)

# --- 4. emit the derived translation unit ---
set(BANNER
"// AUTO-GENERATED — DO NOT EDIT, DO NOT COMMIT.
// Derived mechanically at build time by cmake/generate-tts-upstream.cmake from
// llama.cpp tools/tts/tts.cpp @ ${LLAMA_TAG} (MIT-licensed, the llama.cpp authors).
// Regenerated from the pinned upstream source on every configure; see CLAUDE.md.

")
set(SPEAKER
"
// --- default OuteTTS speaker profile (en_male_1), extracted from upstream main() ---
// `extern const` forces external linkage (a namespace-scope `const` is internal by default),
// so tts_engine.cpp links against these via the `extern` declarations in tts_upstream.h.
extern const std::string jllama_tts_default_audio_text = \"${AUDIO_TEXT}\";
extern const std::string jllama_tts_default_audio_data = R\"(${AUDIO_DATA})\";
")
file(WRITE "${OUT_CPP}" "${BANNER}${PREMAIN}${SPEAKER}")
message(STATUS "generate-tts-upstream: wrote ${OUT_CPP} (from tts.cpp @ ${LLAMA_TAG})")
