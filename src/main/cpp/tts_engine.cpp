// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Native OuteTTS text-to-speech pipeline, single-stream (n_parallel = 1) with the built-in default
// speaker. Loads the TTC (OuteTTS) and CTS (WavTokenizer vocoder) models, builds the OuteTTS prompt,
// generates audio codes, runs the vocoder, and turns the result into a 16-bit WAV (DSP in tts_dsp.hpp).
//
// Known simplification: replace_numbers_with_words is a pass-through, so numeric digits in the input
// are dropped by the alpha-only filter. Spell numbers out in the caller for now.

#include "tts_engine.h"

#include "tts_dsp.hpp"

#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <regex>
#include <string>
#include <vector>

namespace jllama_tts {

namespace {

enum outetts_version { OUTETTS_V0_2, OUTETTS_V0_3 };

// --- OuteTTS prompt helpers ---

std::string process_text(const std::string &text, const outetts_version tts_version) {
    // NOTE: number-to-words and non-English romanization are not ported (see file header).
    std::string processed_text = text;

    std::transform(processed_text.begin(), processed_text.end(), processed_text.begin(), ::tolower);

    processed_text = std::regex_replace(processed_text, std::regex(R"([-_/,\.\\])"), " ");
    processed_text = std::regex_replace(processed_text, std::regex(R"([^a-z\s])"), "");
    processed_text = std::regex_replace(processed_text, std::regex(R"(\s+)"), " ");
    processed_text = std::regex_replace(processed_text, std::regex(R"(^\s+|\s+$)"), "");

    std::string separator = (tts_version == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
    processed_text = std::regex_replace(processed_text, std::regex(R"(\s)"), separator);

    return processed_text;
}

void prompt_add(llama_tokens &prompt, const llama_tokens &tokens) {
    prompt.insert(prompt.end(), tokens.begin(), tokens.end());
}

void prompt_add(llama_tokens &prompt, const llama_vocab *vocab, const std::string &txt, bool add_special,
                bool parse_special) {
    auto tmp = common_tokenize(vocab, txt, add_special, parse_special);
    prompt_add(prompt, tmp);
}

void prompt_init(llama_tokens &prompt, const llama_vocab *vocab) {
    prompt.clear();
    prompt_add(prompt, vocab, "<|im_start|>\n", true, true);
}

std::vector<llama_token> prepare_guide_tokens(const llama_vocab *vocab, const std::string &str,
                                              const outetts_version tts_version) {
    const std::string &delimiter = (tts_version == OUTETTS_V0_3 ? "<|space|>" : "<|text_sep|>");

    std::vector<llama_token> result;
    size_t start = 0;
    size_t end = str.find(delimiter);

    result.push_back(common_tokenize(vocab, "\n", false, true)[0]);

    while (end != std::string::npos) {
        std::string current_word = str.substr(start, end - start);
        auto tmp = common_tokenize(vocab, current_word, false, true);
        result.push_back(tmp[0]);
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    std::string current_word = str.substr(start);
    auto tmp = common_tokenize(vocab, current_word, false, true);
    if (tmp.size() > 0) {
        result.push_back(tmp[0]);
    }
    return result;
}

outetts_version get_tts_version(llama_model *model) {
    const char *chat_template = llama_model_chat_template(model, nullptr);
    if (chat_template && std::string(chat_template) == "outetts-0.3") {
        return OUTETTS_V0_3;
    }
    return OUTETTS_V0_2;
}

// Default OuteTTS speaker profile (en_male_1).
const char *default_audio_text() {
    return "<|text_start|>the<|text_sep|>overall<|text_sep|>package<|text_sep|>from<|text_sep|>just<|text_sep|>two<|"
           "text_sep|>people<|text_sep|>is<|text_sep|>pretty<|text_sep|>remarkable<|text_sep|>sure<|text_sep|>i<|"
           "text_sep|>have<|text_sep|>some<|text_sep|>critiques<|text_sep|>about<|text_sep|>some<|text_sep|>of<|"
           "text_sep|>the<|text_sep|>gameplay<|text_sep|>aspects<|text_sep|>but<|text_sep|>its<|text_sep|>still<|"
           "text_sep|>really<|text_sep|>enjoyable<|text_sep|>and<|text_sep|>it<|text_sep|>looks<|text_sep|>lovely<|"
           "text_sep|>";
}

const char *default_audio_data() {
    return R"(<|audio_start|>
the<|t_0.08|><|code_start|><|257|><|740|><|636|><|913|><|788|><|1703|><|code_end|>
overall<|t_0.36|><|code_start|><|127|><|201|><|191|><|774|><|700|><|532|><|1056|><|557|><|798|><|298|><|1741|><|747|><|1662|><|1617|><|1702|><|1527|><|368|><|1588|><|1049|><|1008|><|1625|><|747|><|1576|><|728|><|1019|><|1696|><|1765|><|code_end|>
package<|t_0.56|><|code_start|><|935|><|584|><|1319|><|627|><|1016|><|1491|><|1344|><|1117|><|1526|><|1040|><|239|><|1435|><|951|><|498|><|723|><|1180|><|535|><|789|><|1649|><|1637|><|78|><|465|><|1668|><|901|><|595|><|1675|><|117|><|1009|><|1667|><|320|><|840|><|79|><|507|><|1762|><|1508|><|1228|><|1768|><|802|><|1450|><|1457|><|232|><|639|><|code_end|>
from<|t_0.19|><|code_start|><|604|><|782|><|1682|><|872|><|1532|><|1600|><|1036|><|1761|><|647|><|1554|><|1371|><|653|><|1595|><|950|><|code_end|>
just<|t_0.25|><|code_start|><|1782|><|1670|><|317|><|786|><|1748|><|631|><|599|><|1155|><|1364|><|1524|><|36|><|1591|><|889|><|1535|><|541|><|440|><|1532|><|50|><|870|><|code_end|>
two<|t_0.24|><|code_start|><|1681|><|1510|><|673|><|799|><|805|><|1342|><|330|><|519|><|62|><|640|><|1138|><|565|><|1552|><|1497|><|1552|><|572|><|1715|><|1732|><|code_end|>
people<|t_0.39|><|code_start|><|593|><|274|><|136|><|740|><|691|><|633|><|1484|><|1061|><|1138|><|1485|><|344|><|428|><|397|><|1562|><|645|><|917|><|1035|><|1449|><|1669|><|487|><|442|><|1484|><|1329|><|1832|><|1704|><|600|><|761|><|653|><|269|><|code_end|>
is<|t_0.16|><|code_start|><|566|><|583|><|1755|><|646|><|1337|><|709|><|802|><|1008|><|485|><|1583|><|652|><|10|><|code_end|>
pretty<|t_0.32|><|code_start|><|1818|><|1747|><|692|><|733|><|1010|><|534|><|406|><|1697|><|1053|><|1521|><|1355|><|1274|><|816|><|1398|><|211|><|1218|><|817|><|1472|><|1703|><|686|><|13|><|822|><|445|><|1068|><|code_end|>
remarkable<|t_0.68|><|code_start|><|230|><|1048|><|1705|><|355|><|706|><|1149|><|1535|><|1787|><|1356|><|1396|><|835|><|1583|><|486|><|1249|><|286|><|937|><|1076|><|1150|><|614|><|42|><|1058|><|705|><|681|><|798|><|934|><|490|><|514|><|1399|><|572|><|1446|><|1703|><|1346|><|1040|><|1426|><|1304|><|664|><|171|><|1530|><|625|><|64|><|1708|><|1830|><|1030|><|443|><|1509|><|1063|><|1605|><|1785|><|721|><|1440|><|923|><|code_end|>
i<|t_0.08|><|code_start|><|123|><|439|><|1074|><|705|><|1799|><|637|><|code_end|>
have<|t_0.16|><|code_start|><|1509|><|599|><|518|><|1170|><|552|><|1029|><|1267|><|864|><|419|><|143|><|1061|><|0|><|code_end|>
some<|t_0.16|><|code_start|><|619|><|400|><|1270|><|62|><|1370|><|1832|><|917|><|1661|><|167|><|269|><|1366|><|1508|><|code_end|>
of<|t_0.07|><|code_start|><|197|><|716|><|1039|><|1662|><|64|><|code_end|>
the<|t_0.08|><|code_start|><|1811|><|1568|><|569|><|886|><|1025|><|1374|><|code_end|>
looks<|t_0.27|><|code_start|><|1281|><|1266|><|1755|><|572|><|248|><|1751|><|1257|><|695|><|1380|><|457|><|659|><|585|><|1315|><|1105|><|1776|><|736|><|24|><|736|><|654|><|1027|><|code_end|>
lovely<|t_0.56|><|code_start|><|634|><|596|><|1766|><|1556|><|1306|><|1285|><|1481|><|1721|><|1123|><|438|><|1246|><|1251|><|795|><|659|><|1381|><|1658|><|217|><|1772|><|562|><|952|><|107|><|1129|><|1112|><|467|><|550|><|1079|><|840|><|1615|><|1469|><|1380|><|168|><|917|><|836|><|1827|><|437|><|583|><|67|><|595|><|1087|><|1646|><|1493|><|1677|><|code_end|>)";
}

} // namespace

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
    engine->tts_version = get_tts_version(engine->model_ttc);

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
    std::string audio_text = default_audio_text();
    std::string audio_data = default_audio_data();
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

    // Keep only audio codec tokens (151672..155772) and rebase to codec ids.
    codes.erase(std::remove_if(codes.begin(), codes.end(), [](llama_token t) { return t < 151672 || t > 155772; }),
                codes.end());
    for (auto &token : codes) {
        token -= 151672;
    }
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

    const int n_embd = llama_model_n_embd_out(engine->model_cts);
    const float *embd = llama_get_embeddings(engine->ctx_cts);
    std::vector<float> audio = embd_to_audio(embd, n_codes, n_embd, engine->n_threads);
    llama_batch_free(cts_batch);

    // Zero the first 0.25 s (suppresses a leading click).
    const int n_sr = 24000;
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
