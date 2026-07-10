// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Native OuteTTS text-to-speech pipeline, single-stream (n_parallel = 1) with the built-in default
// speaker. Loads the TTC (OuteTTS) and CTS (WavTokenizer vocoder) models, builds the OuteTTS prompt,
// generates audio codes, runs the vocoder, and turns the result into a 16-bit WAV.
//
// The OuteTTS DSP / prompt / text helpers and the default-speaker profile are NOT reimplemented here:
// they are derived at build time from the pinned upstream llama.cpp tools/tts/tts.cpp (see
// tts_upstream.h + cmake/generate-tts-upstream.cmake) and called directly. This file is only the
// orchestration that drives those helpers across the two models. The in-memory WAV writer is ours
// (tts_wav.hpp).

#include "tts_engine.h"

#include "tts_upstream.h" // embd_to_audio, process_text, prompt_*, prepare_guide_tokens, get_tts_version, default speaker
#include "tts_wav.hpp"    // pcm_to_wav16_bytes

#include "common.h"
#include "llama.h"
#include "sampling.h"

// Full json definition: tts_upstream.h only forward-declares nlohmann::ordered_json (keeping the heavy
// header out of the shared interface), but this TU constructs the empty-object speaker argument for
// get_tts_version(), which needs the complete type.
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <regex>
#include <string>
#include <vector>

namespace jllama_tts {

struct tts_engine {
    common_init_result_ptr init_ttc;
    common_init_result_ptr init_cts;
    llama_model *model_ttc = nullptr;
    llama_context *ctx_ttc = nullptr;
    llama_model *model_cts = nullptr;
    llama_context *ctx_cts = nullptr;
    const llama_vocab *vocab = nullptr;
    outetts_version tts_version = OUTETTS_V0_2;
    int n_threads = 4;
    // Serializes engine_synthesize: it drives llama_decode/llama_encode on the shared
    // ctx_ttc/ctx_cts contexts, so two threads on one engine would race (M5).
    std::mutex synthesize_mutex;
};

tts_engine *engine_init(const std::string &ttc_model_path, const std::string &cts_model_path, int n_gpu_layers,
                        int n_threads, std::string &err) {
    llama_backend_init();

    auto engine = new tts_engine();
    engine->n_threads = n_threads > 0 ? n_threads : 4;

    common_params params;
    params.n_ctx = 8192;
    params.n_batch = 8192;
    params.n_gpu_layers = n_gpu_layers;
    params.cpuparams.n_threads = engine->n_threads;
    params.sampling.top_k = 4;
    params.sampling.samplers = {COMMON_SAMPLER_TYPE_TOP_K};

    // Text-to-codes (TTC) model.
    params.model.path = ttc_model_path;
    engine->init_ttc = common_init_from_params(params);
    engine->model_ttc = engine->init_ttc ? engine->init_ttc->model() : nullptr;
    engine->ctx_ttc = engine->init_ttc ? engine->init_ttc->context() : nullptr;
    if (engine->model_ttc == nullptr || engine->ctx_ttc == nullptr) {
        err = "failed to load TTC (text-to-codes) model: " + ttc_model_path;
        engine_free(engine);
        return nullptr;
    }
    engine->vocab = llama_model_get_vocab(engine->model_ttc);
    // Explicit empty-object speaker: tts_upstream.h declares no default (it forward-declares json), so
    // the default lives only in the generated TU. We always use the built-in default speaker profile.
    engine->tts_version = get_tts_version(engine->model_ttc, nlohmann::ordered_json::object());

    // Codes-to-speech (CTS) vocoder, loaded in embedding mode.
    params.model.path = cts_model_path;
    params.embedding = true;
    params.n_ubatch = params.n_batch;
    engine->init_cts = common_init_from_params(params);
    engine->model_cts = engine->init_cts ? engine->init_cts->model() : nullptr;
    engine->ctx_cts = engine->init_cts ? engine->init_cts->context() : nullptr;
    if (engine->model_cts == nullptr || engine->ctx_cts == nullptr) {
        err = "failed to load CTS (vocoder) model: " + cts_model_path;
        engine_free(engine);
        return nullptr;
    }

    return engine;
}

bool engine_synthesize(tts_engine *engine, const std::string &text, int n_predict, int top_k, uint32_t seed,
                       std::vector<uint8_t> &out_wav, std::string &err) {
    if (engine == nullptr) {
        err = "engine is null";
        return false;
    }
    // Serialize against concurrent calls on the same engine (M5).
    std::lock_guard<std::mutex> engine_lock(engine->synthesize_mutex);
    const llama_vocab *vocab = engine->vocab;

    common_params_sampling sparams;
    sparams.top_k = top_k > 0 ? top_k : 4;
    sparams.seed = seed;
    sparams.samplers = {COMMON_SAMPLER_TYPE_TOP_K};
    common_sampler *smpl = common_sampler_init(engine->model_ttc, sparams);
    if (smpl == nullptr) {
        err = "failed to init sampler";
        return false;
    }

    // Build the OuteTTS prompt: speaker text + processed input + speaker audio codes.
    std::string audio_text = jllama_tts_default_audio_text;
    std::string audio_data = jllama_tts_default_audio_data;
    if (engine->tts_version == OUTETTS_V0_3) {
        audio_text = std::regex_replace(audio_text, std::regex(R"(<\|text_sep\|>)"), "<|space|>");
        audio_data = std::regex_replace(audio_data, std::regex(R"(<\|code_start\|>)"), "");
        audio_data = std::regex_replace(audio_data, std::regex(R"(<\|code_end\|>)"), "<|space|>");
    }

    llama_tokens prompt_inp;
    prompt_init(prompt_inp, vocab);
    prompt_add(prompt_inp, vocab, audio_text, false, true);

    std::string prompt_clean = process_text(text, engine->tts_version);
    std::vector<llama_token> guide_tokens = prepare_guide_tokens(vocab, prompt_clean, engine->tts_version);
    prompt_add(prompt_inp, vocab, prompt_clean, false, true);
    prompt_add(prompt_inp, vocab, "<|text_end|>\n", false, true);
    prompt_add(prompt_inp, vocab, audio_data, false, true);

    std::vector<llama_token> codes;

    // Decode the prompt (logits for the last token only).
    llama_batch batch = llama_batch_init((int32_t)prompt_inp.size(), 0, 1);
    for (size_t i = 0; i < prompt_inp.size(); ++i) {
        common_batch_add(batch, prompt_inp[i], (llama_pos)i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    if (llama_decode(engine->ctx_ttc, batch) != 0) {
        llama_batch_free(batch);
        common_sampler_free(smpl);
        err = "llama_decode failed on the TTS prompt";
        return false;
    }
    llama_synchronize(engine->ctx_ttc);

    // Generate audio codec tokens.
    int n_past = batch.n_tokens;
    int n_decode = 0;
    int i_batch = batch.n_tokens - 1;
    bool next_token_uses_guide_token = true;
    const int predict = n_predict > 0 ? n_predict : 4096;

    while (n_decode <= predict) {
        common_batch_clear(batch);

        llama_token new_token_id = common_sampler_sample(smpl, engine->ctx_ttc, i_batch);
        if (!guide_tokens.empty() && next_token_uses_guide_token && !llama_vocab_is_control(vocab, new_token_id) &&
            !llama_vocab_is_eog(vocab, new_token_id)) {
            new_token_id = guide_tokens.front();
            guide_tokens.erase(guide_tokens.begin());
        }
        next_token_uses_guide_token = (new_token_id == 198);

        common_sampler_accept(smpl, new_token_id, true);
        codes.push_back(new_token_id);

        if (llama_vocab_is_eog(vocab, new_token_id) || n_decode == predict) {
            break;
        }

        i_batch = batch.n_tokens;
        common_batch_add(batch, new_token_id, n_past, {0}, true);

        n_decode += 1;
        n_past += 1;
        if (llama_decode(engine->ctx_ttc, batch) != 0) {
            llama_batch_free(batch);
            common_sampler_free(smpl);
            err = "llama_decode failed during code generation";
            return false;
        }
    }
    llama_batch_free(batch);
    common_sampler_free(smpl);

    // Keep only audio codec tokens and rebase to 0-based codec ids. The window is identical
    // for OuteTTS V0_2 and V0_3 (verified against upstream tools/tts/tts.cpp, see M3 in
    // docs/plan-bugs-and-issues.md) — delegated to the pure filter_ouetts_codec_tokens helper.
    filter_ouetts_codec_tokens(codes);
    if (codes.empty()) {
        err = "no audio codes were generated";
        return false;
    }

    // Run the vocoder over the codes and read the output embeddings.
    const int n_codes = (int)codes.size();
    llama_batch cts_batch = llama_batch_init(n_codes, 0, 1);
    for (size_t i = 0; i < codes.size(); ++i) {
        common_batch_add(cts_batch, codes[i], (llama_pos)i, {0}, true);
    }
    if (llama_encode(engine->ctx_cts, cts_batch) != 0) {
        llama_batch_free(cts_batch);
        err = "llama_encode (vocoder) failed";
        return false;
    }
    llama_synchronize(engine->ctx_cts);

    // llama_model_n_embd_out (not llama_model_n_embd): read the vocoder's OUTPUT embedding width, which
    // is what llama_get_embeddings returns here. This matches upstream tts.cpp, which also queries
    // llama_model_n_embd_out at this step.
    const int n_embd = llama_model_n_embd_out(engine->model_cts);
    const float *embd = llama_get_embeddings(engine->ctx_cts);
    std::vector<float> audio = embd_to_audio(embd, n_codes, n_embd, engine->n_threads);
    llama_batch_free(cts_batch);

    // 24 kHz mono — the OuteTTS / WavTokenizer output rate.
    const int n_sr = 24000;
    // Zero the first 0.25 s, mirroring upstream tts.cpp's post-vocoder cleanup (it suppresses a leading
    // click). The `&& i < audio.size()` guard is ours: it keeps the loop in-bounds for clips shorter
    // than 0.25 s, where upstream's fixed 24000/4 bound would read past the buffer.
    for (int i = 0; i < n_sr / 4 && i < (int)audio.size(); ++i) {
        audio[i] = 0.0f;
    }

    out_wav = pcm_to_wav16_bytes(audio, n_sr);
    return true;
}

void engine_free(tts_engine *engine) {
    if (engine == nullptr) {
        return;
    }
    // init_ttc / init_cts own the models + contexts and free them on destruction.
    delete engine;
}

} // namespace jllama_tts
