#pragma once

#include "base64.hpp"
#include "common.h"
#include "llama.h"
#include "llama-model.h"
#include "log.h"
#include "minja/chat-template.hpp"
#include "minja/minja.hpp"
#include "clip.h"
#include "llava.h"
#include "v_chat.hpp"
#include "v_clip.hpp"
#include "sampling.h"
#include "ngram-cache.h"


#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif
// increase max payload length to allow use of larger context size
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 1048576
// #include "httplib.h"

// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "nlohmann/json.hpp"

#include "chat.h"
#include "llama-chat.h"

#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo"

using json = nlohmann::ordered_json;

#define SLT_INF(slot, fmt, ...)                                                                                        \
    LOG_INF("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, (slot).id_task, __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...)                                                                                        \
    LOG_WRN("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, (slot).id_task, __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...)                                                                                        \
    LOG_ERR("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, (slot).id_task, __VA_ARGS__)
#define SLT_DBG(slot, fmt, ...)                                                                                        \
    LOG_DBG("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, (slot).id_task, __VA_ARGS__)

#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)



// Trim from start (left)
static inline std::string ltrim(const std::string &s) {
    std::string copy = s;
    copy.erase(copy.begin(), std::find_if(copy.begin(), copy.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    return copy;
}

// Trim from end (right)
static inline std::string rtrim(const std::string &s) {
    std::string copy = s;
    copy.erase(std::find_if(copy.rbegin(), copy.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), copy.end());
    return copy;
}

// Trim from both ends
static inline std::string trim(const std::string &s) {
    return ltrim(rtrim(s));
}

const static std::string build_info("b" + std::to_string(LLAMA_BUILD_NUMBER) + "-" + LLAMA_COMMIT);

//
// tokenizer and input processing utils
//
template <typename T> 

static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const &) {
            LOG_WRN("Wrong type supplied for parameter '%s'. Expected '%s', using default value\n", key.c_str(), json(default_value).type_name());
            return default_value;
        }
    } else {
        return default_value;
    }
}
static bool json_is_array_of_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (!e.is_number_integer()) {
                return false;
            }
        }
        return true;
    }
    return false;
}

// is array having BOTH numbers & strings?
static bool json_is_array_of_mixed_numbers_strings(const json & data) {
    bool seen_string = false;
    bool seen_number = false;
    if (data.is_array()) {
        for (const auto & e : data) {
            seen_string |= e.is_string();
            seen_number |= e.is_number_integer();
            if (seen_number && seen_string) {
                return true;
            }
        }
    }
    return false;
}

// get value by path(key1 / key2)
static json json_get_nested_values(const std::vector<std::string> & paths, const json & js) {
    json result = json::object();

    for (const std::string & path : paths) {
        json current = js;
        const auto keys = string_split<std::string>(path, /*separator*/ '/');
        bool valid_path = true;
        for (const std::string & k : keys) {
            if (valid_path && current.is_object() && current.contains(k)) {
                current = current[k];
            } else {
                valid_path = false;
            }
        }
        if (valid_path) {
            result[path] = current;
        }
    }
    return result;
}

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
static llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
    // If `add_bos` is true, we only add BOS, when json_prompt is a string,
    // or the first element of the json_prompt array is a string.
    llama_tokens prompt_tokens;

    if (json_prompt.is_array()) {
        bool first = true;
        for (const auto & p : json_prompt) {
            if (p.is_string()) {
                auto s = p.template get<std::string>();

                llama_tokens p;
                if (first) {
                    p = common_tokenize(vocab, s, add_special, parse_special);
                    first = false;
                } else {
                    p = common_tokenize(vocab, s, false, parse_special);
                }

                prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
            } else {
                if (first) {
                    first = false;
                }

                prompt_tokens.push_back(p.template get<llama_token>());
            }
        }
    } else {
        auto s = json_prompt.template get<std::string>();
        prompt_tokens = common_tokenize(vocab, s, add_special, parse_special);
    }

    return prompt_tokens;
}

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56]]
 */
static std::vector<llama_tokens> tokenize_input_prompts(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
    std::vector<llama_tokens> result;
    if (json_prompt.is_string() || json_is_array_of_mixed_numbers_strings(json_prompt)) {
        // string or mixed
        result.push_back(tokenize_mixed(vocab, json_prompt, add_special, parse_special));
    } else if (json_is_array_of_numbers(json_prompt)) {
        // array of tokens
        result.push_back(json_prompt.get<llama_tokens>());
    } else if (json_prompt.is_array()) {
        // array of prompts
        result.reserve(json_prompt.size());
        for (const auto & p : json_prompt) {
            if (p.is_string() || json_is_array_of_mixed_numbers_strings(p)) {
                result.push_back(tokenize_mixed(vocab, p, add_special, parse_special));
            } else if (json_is_array_of_numbers(p)) {
                // array of tokens
                result.push_back(p.get<llama_tokens>());
            } else {
                throw std::runtime_error("element of \"prompt\" must be a string, an list of tokens, or a list of mixed strings & tokens");
            }
        }
    } else {
        throw std::runtime_error("\"prompt\" must be a string, an list of tokens, a list of mixed strings & tokens, or a list of prompts");
    }
    if (result.empty()) {
        throw std::runtime_error("\"prompt\" must not be empty");
    }
    return result;
}

// return the last index of character that can form a valid string
// if the last character is potentially cut in half, return the index before the cut
// if validate_utf8(text) == text.size(), then the whole text is valid utf8
static size_t validate_utf8(const std::string& text) {
    size_t len = text.size();
    if (len == 0) return 0;

    // Check the last few bytes to see if a multi-byte character is cut off
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = text[len - i];
        // Check for start of a multi-byte sequence from the end
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character start: 110xxxxx
            // Needs at least 2 bytes
            if (i < 2) return len - i;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character start: 1110xxxx
            // Needs at least 3 bytes
            if (i < 3) return len - i;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character start: 11110xxx
            // Needs at least 4 bytes
            if (i < 4) return len - i;
        }
    }

    // If no cut-off multi-byte character is found, return full length
    return len;
}

//
// template utils
//

// format rerank task: [BOS]query[EOS][SEP]doc[EOS]
static llama_tokens format_rerank(const struct llama_vocab * vocab, const llama_tokens & query, const llama_tokens & doc) {
    llama_tokens result;

    result.reserve(doc.size() + query.size() + 4);
    result.push_back(llama_vocab_bos(vocab));
    result.insert(result.end(), query.begin(), query.end());
    result.push_back(llama_vocab_eos(vocab));
    result.push_back(llama_vocab_sep(vocab));
    result.insert(result.end(), doc.begin(), doc.end());
    result.push_back(llama_vocab_eos(vocab));

    return result;
}

// format infill task
static llama_tokens format_infill(
        const llama_vocab * vocab,
        const json & input_prefix,
        const json & input_suffix,
        const json & input_extra,
        const int n_batch,
        const int n_predict,
        const int n_ctx,
        const bool spm_infill,
        const llama_tokens & tokens_prompt
    ) {
    // TODO: optimize this block by reducing memory allocations and movement

    // use FIM repo-level pattern:
    // ref: https://arxiv.org/pdf/2409.12186
    //
    // [FIM_REP]myproject
    // [FIM_SEP]filename0
    // extra chunk 0
    // [FIM_SEP]filename1
    // extra chunk 1
    // ...
    // [FIM_SEP]filename
    // [FIM_PRE]prefix[FIM_SUF]suffix[FIM_MID]prompt
    //
    llama_tokens extra_tokens;
    extra_tokens.reserve(n_ctx);

    auto tokens_prefix = tokenize_mixed(vocab, input_prefix, false, false);
    auto tokens_suffix = tokenize_mixed(vocab, input_suffix, false, false);

    if (llama_vocab_fim_rep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: make project name an input
        static const auto k_fim_repo = common_tokenize(vocab, "myproject\n", false, false);

        extra_tokens.push_back(llama_vocab_fim_rep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_repo.begin(), k_fim_repo.end());
    }
    for (const auto & chunk : input_extra) {
        // { "text": string, "filename": string }
        const std::string text     = json_value(chunk, "text",     std::string());
        const std::string filename = json_value(chunk, "filename", std::string("tmp"));

        if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
            const auto k_fim_file = common_tokenize(vocab, filename + "\n", false, false);

            extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
            extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
        } else {
            // chunk separator in binary form to avoid confusing the AI
            static const char k_chunk_prefix_str[] = {0x0a, 0x0a, 0x2d, 0x2d, 0x2d, 0x20, 0x73, 0x6e, 0x69, 0x70, 0x70, 0x65, 0x74, 0x20, 0x2d, 0x2d, 0x2d, 0x0a, 0x0a, 0x00};
            static const auto k_chunk_prefix_tokens = common_tokenize(vocab, k_chunk_prefix_str, false, false);

            extra_tokens.insert(extra_tokens.end(), k_chunk_prefix_tokens.begin(), k_chunk_prefix_tokens.end());
        }

        const auto chunk_tokens = common_tokenize(vocab, text, false, false);
        extra_tokens.insert(extra_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    }

    if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: current filename
        static const auto k_fim_file = common_tokenize(vocab, "filename\n", false, false);

        extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
    }

    // for now pick FIM context to fit in a batch (ratio prefix:suffix = 3:1, TODO: configurable?)
    const int n_prefix_take = std::min<int>(tokens_prefix.size(),                3*(n_batch/4));
    const int n_suffix_take = std::min<int>(tokens_suffix.size(), std::max<int>(0, (n_batch/4) - (2 + tokens_prompt.size())));

    SRV_DBG("n_prefix_take = %d, n_suffix_take = %d, total = %d\n", n_prefix_take, n_suffix_take, (n_prefix_take + n_suffix_take));

    // fill the rest of the context with extra chunks
    const int n_extra_take = std::min<int>(std::max<int>(0, n_ctx - (n_batch) - 2*n_predict), extra_tokens.size());

    tokens_prefix.erase(tokens_prefix.begin(), tokens_prefix.begin() + tokens_prefix.size() - n_prefix_take);
    tokens_suffix.resize(n_suffix_take);

    tokens_prefix.insert(tokens_prefix.begin(), llama_vocab_fim_pre(vocab));
    tokens_prefix.insert(tokens_prefix.end(),   tokens_prompt.begin(), tokens_prompt.end());
    tokens_suffix.insert(tokens_suffix.begin(), llama_vocab_fim_suf(vocab));

    auto embd_inp = spm_infill ? tokens_suffix : tokens_prefix;
    auto embd_end = spm_infill ? tokens_prefix : tokens_suffix;

    if (llama_vocab_get_add_bos(vocab)) {
        embd_inp.insert(embd_inp.begin(), llama_vocab_bos(vocab));
    }

    SRV_DBG("extra: n_ctx = %d, n_extra_take = %d, n_extra = %d\n", n_ctx, n_extra_take, (int) extra_tokens.size());

    // put the extra context before the FIM prefix
    embd_inp.insert(embd_inp.begin(), extra_tokens.end() - n_extra_take, extra_tokens.end());

    embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());
    embd_inp.push_back(llama_vocab_fim_mid(vocab));

    return embd_inp;
}

//
// base64 utils (TODO: move to common in the future)
//

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(uint8_t c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline std::vector<uint8_t> base64_decode(const std::string & encoded_string) {
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    std::vector<uint8_t> ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }

            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}

static inline const std::string base64_encode(const unsigned char *input, size_t length) {
    std::string output;
    output.reserve(length);

    auto val = 0, valb = -6;
    for (size_t i = 0; i < length; ++i) {
        val = (val << 8) + static_cast<uint8_t>(input[i]);
        valb += 8;
        while (valb >= 0) {
            output.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }

    if (valb > -6) {
        output.push_back(base64_chars[((val << 8) >> valb) & 0x3F]);
    }

    while (output.size() % 4) {
        output.push_back('=');
    }

    return output;
}

//
// random string / id
//

static std::string random_string() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

static std::string gen_chatcmplid() {
    return "chatcmpl-" + random_string();
}

static std::string gen_tool_call_id() {
    return random_string();
}

static std::string gen_callid() {
    return "call-" + random_string();
}

//
// other common utils
//

static bool starts_with(const std::string &str, const std::string &prefix) {
    return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
}

static bool ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string &stop, const std::string &text) {
    if (!text.empty() && !stop.empty()) {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--) {
            if (stop[char_index] == text_last_char) {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial)) {
                    return text.size() - char_index - 1;
                }
            }
        }
    }

    return std::string::npos;
}

// TODO: reuse llama_detokenize
template <class Iter>
static std::string tokens_to_str(llama_context * ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += common_token_to_piece(ctx, *begin);
    }

    return ret;
}

// format incomplete utf-8 multibyte character for output
static std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token) {
    std::string out = token == LLAMA_TOKEN_NULL ? "" : common_token_to_piece(ctx, token);

    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80) {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }

    return out;
}

struct completion_token_output {
    struct token_prob {
        llama_token tok;
        float prob;
    };

    std::vector<llama_token> toks;
    std::vector<float> probs;
    std::vector<std::vector<token_prob>> top_probs;
    std::string text_to_send;
};

// convert a vector of completion_token_output to json
static json probs_vector_to_json(const llama_context *ctx, const std::vector<completion_token_output> &probs, const bool oaicompat_completion = false, const bool oaicompat_completion_chat = false) {
    if (oaicompat_completion) {
        if (oaicompat_completion_chat) {
            json content = json::array();

            for (const auto &p : probs) {
                for (size_t i = 0; i < p.toks.size(); i++) {
                    const llama_token id    = p.toks[i];
                    const std::string token = tokens_to_output_formatted_string(ctx, id);
                    float token_logprob     = p.probs[i] == 0.0f ? std::numeric_limits<float>::lowest() : std::log(p.probs[i]);
                    std::vector<unsigned char> token_bytes(token.begin(), token.end());
                    json token_top_logprobs = json::array();
                    for (const auto &tp : p.top_probs[i]) {
                        const llama_token tp_id    = tp.tok;
                        const std::string tp_token = tokens_to_output_formatted_string(ctx, tp_id);
                        float tp_token_logprob     = tp.prob == 0.0f ? std::numeric_limits<float>::lowest() : std::log(tp.prob);
                        std::vector<unsigned char> tp_token_bytes(tp_token.begin(), tp_token.end());
                        token_top_logprobs.push_back(json{
                            {"token", tp_token},
                            {"logprob", tp_token_logprob},
                            {"bytes", tp_token_bytes},
                        });
                    }

                    content.push_back(json{
                        {"token", token},
                        {"logprob", token_logprob},
                        {"bytes", token_bytes},
                        {"top_logprobs", token_top_logprobs},
                    });
                }
            }

            return json{{"content", content}};
        } else {
            json token_logprobs = json::array();
            json tokens         = json::array();
            json top_logprobs   = json::array();

            for (const auto &p : probs) {
                for (size_t i = 0; i < p.toks.size(); i++) {
                    const llama_token id    = p.toks[i];
                    const std::string token = tokens_to_output_formatted_string(ctx, id);
                    float token_logprob     = p.probs[i] == 0.0f ? std::numeric_limits<float>::lowest() : std::log(p.probs[i]);
                    json token_top_logprobs;
                    for (const auto &tp : p.top_probs[i]) {
                        const llama_token tp_id      = tp.tok;
                        const std::string tp_token   = tokens_to_output_formatted_string(ctx, tp_id);
                        float tp_token_logprob       = tp.prob == 0.0f ? std::numeric_limits<float>::lowest() : std::log(tp.prob);
                        token_top_logprobs[tp_token] = tp_token_logprob;
                    }

                    tokens.push_back(token);
                    token_logprobs.push_back(token_logprob);
                    top_logprobs.push_back(token_top_logprobs);
                }
            }

            return json{
                {"tokens", tokens},
                {"token_logprobs", token_logprobs},
                {"top_logprobs", top_logprobs},
            };
        }
    }

    json out = json::array();

    for (const auto &p : probs) {
        for (size_t i = 0; i < p.toks.size(); i++) {
            const llama_token id    = p.toks[i];
            const std::string token = tokens_to_output_formatted_string(ctx, id);
            float token_prob        = p.probs[i];
            std::vector<unsigned char> token_bytes(token.begin(), token.end());
            json token_top_probs = json::array();
            for (const auto &tp : p.top_probs[i]) {
                const llama_token tp_id    = tp.tok;
                const std::string tp_token = tokens_to_output_formatted_string(ctx, tp_id);
                float tp_token_prob        = tp.prob;
                std::vector<unsigned char> tp_token_bytes(tp_token.begin(), tp_token.end());
                token_top_probs.push_back(json{
                    {"id", tp_id},
                    {"token", tp_token},
                    {"prob", tp_token_prob},
                    {"bytes", tp_token_bytes},
                });
            }

            out.push_back(json{
                {"id", id},
                {"token", token},
                {"prob", token_prob},
                {"bytes", token_bytes},
                {"top_probs", token_top_probs},
            });
        }
    }

    return out;
}

//static bool server_sent_event(httplib::DataSink & sink, const char * event, const json & data) {
//    const std::string str =
//        std::string(event) + ": " +
//        data.dump(-1, ' ', false, json::error_handler_t::replace) +
//        "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).
//
//    LOG_DBG("data stream, to_send: %s", str.c_str());
//
//    return sink.write(str.c_str(), str.size());
//}

//
// OAI utils
//

static json oaicompat_completion_params_parse(const json & body) {
    json llama_params;

    if (!body.contains("prompt")) {
        throw std::runtime_error("\"prompt\" is required");
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    // Handle "n" field
    int n_choices = json_value(body, "n", 1);
    if (n_choices != 1) {
        throw std::runtime_error("Only one completion choice is allowed");
    }

    // Handle "echo" field
    if (json_value(body, "echo", false)) {
        throw std::runtime_error("Only no echo is supported");
    }

    // Params supported by OAI but unsupported by llama.cpp
    static const std::vector<std::string> unsupported_params { "best_of", "suffix" };
    for (const auto & param : unsupported_params) {
        if (body.contains(param)) {
            throw std::runtime_error("Unsupported param: " + param);
        }
    }

    // Copy remaining properties to llama_params
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

static json oaicompat_completion_params_parse(
    const json & body, /* openai api json semantics */
    bool use_jinja,
    common_reasoning_format reasoning_format,
    const struct common_chat_templates * tmpls)
{
    json llama_params;

    auto tools = json_value(body, "tools", json());
    auto stream = json_value(body, "stream", false);

    if (tools.is_array() && !tools.empty()) {
        if (stream) {
            throw std::runtime_error("Cannot use tools with stream");
        }
        if (!use_jinja) {
            throw std::runtime_error("tools param requires --jinja flag");
        }
    }
    if (!use_jinja) {
        if (body.contains("tool_choice") && !body.at("tool_choice").is_null()) {
            throw std::runtime_error("Unsupported param: tool_choice");
        }
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    auto json_schema = json_value(body, "json_schema", json());
    auto grammar = json_value(body, "grammar", std::string());
    if (!json_schema.is_null() && !grammar.empty()) {
        throw std::runtime_error("Cannot use both json_schema and grammar");
    }

    // Handle "response_format" field
    if (body.contains("response_format")) {
        json response_format      = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            json_schema = json_value(response_format, "schema", json::object());
        } else if (response_type == "json_schema") {
            auto schema_wrapper = json_value(response_format, "json_schema", json::object());
            json_schema = json_value(schema_wrapper, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::runtime_error("response_format type must be one of \"text\" or \"json_object\", but got: " + response_type);
        }
    }

    common_chat_templates_inputs inputs;
    inputs.messages              = common_chat_msgs_parse_oaicompat(body.at("messages"));
    inputs.tools                 = common_chat_tools_parse_oaicompat(tools);
    inputs.tool_choice           = common_chat_tool_choice_parse_oaicompat(json_value(body, "tool_choice", std::string("auto")));
    inputs.json_schema           = json_schema.is_null() ? "" : json_schema.dump();
    inputs.grammar               = grammar;
    inputs.add_generation_prompt = json_value(body, "add_generation_prompt", true);
    inputs.use_jinja             = use_jinja;
    inputs.parallel_tool_calls   = json_value(body, "parallel_tool_calls", false);
    inputs.extract_reasoning     = reasoning_format != COMMON_REASONING_FORMAT_NONE;
    inputs.add_generation_prompt = json_value(body, "add_generation_prompt", true);
    if (!inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && body.contains("grammar")) {
        throw std::runtime_error("Cannot use custom grammar constraints with tools.");
    }

    // Apply chat template to the list of messages
    auto chat_params = common_chat_templates_apply(tmpls, inputs);

    llama_params["chat_format"]      = static_cast<int>(chat_params.format);
    llama_params["prompt"]           = chat_params.prompt;
    if (!chat_params.grammar.empty()) {
        llama_params["grammar"] = chat_params.grammar;
    }
    llama_params["grammar_lazy"]     = chat_params.grammar_lazy;
    auto grammar_triggers = json::array();
    for (const auto & trigger : chat_params.grammar_triggers) {
        grammar_triggers.push_back(trigger.to_json<json>());
    }
    llama_params["grammar_triggers"] = grammar_triggers;
    llama_params["preserved_tokens"] = chat_params.preserved_tokens;
    for (const auto & stop : chat_params.additional_stops) {
        llama_params["stop"].push_back(stop);
    }

    // Handle "n" field
    int n_choices = json_value(body, "n", 1);
    if (n_choices != 1) {
        throw std::runtime_error("Only one completion choice is allowed");
    }

    // Handle "logprobs" field
    // TODO: The response format of this option is not yet OAI-compatible, but seems like no one really using it; We may need to fix it in the future
    if (json_value(body, "logprobs", false)) {
        llama_params["n_probs"] = json_value(body, "top_logprobs", 20);
    } else if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        throw std::runtime_error("top_logprobs requires logprobs to be set to true");
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", ... via OAI endpoint.
    // See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

static json format_embeddings_response_oaicompat(const json & request, const json & embeddings, bool use_base64 = false) {
    json data = json::array();
    int32_t n_tokens = 0;
    int i = 0;
    for (const auto & elem : embeddings) {
        json embedding_obj;

        if (use_base64) {
            const auto& vec = json_value(elem, "embedding", json::array()).get<std::vector<float>>();
            const char* data_ptr = reinterpret_cast<const char*>(vec.data());
            size_t data_size = vec.size() * sizeof(float);
            embedding_obj = {
                {"embedding", base64::encode(data_ptr, data_size)},
                {"index", i++},
                {"object", "embedding"},
                {"encoding_format", "base64"}
            };
        } else {
            embedding_obj = {
                {"embedding", json_value(elem, "embedding", json::array())},
                {"index", i++},
                {"object", "embedding"}
            };
        }
        data.push_back(embedding_obj);

        n_tokens += json_value(elem, "tokens_evaluated", 0);
    }

    json res = json {
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"usage", json {
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"data", data}
    };

    return res;
}

static json format_response_rerank(
        const json & request,
        const json & ranks,
        bool is_tei_format,
        std::vector<std::string> & texts) {
    json res;
    if (is_tei_format) {
        // TEI response format
        res = json::array();
        bool return_text = json_value(request, "return_text", false);
        for (const auto & rank : ranks) {
            int index = json_value(rank, "index", 0);
            json elem = json{
                {"index", index},
                {"score", json_value(rank, "score", 0.0)},
            };
            if (return_text) {
                elem["text"] = std::move(texts[index]);
            }
            res.push_back(elem);
        }
    } else {
        // Jina response format
        json results = json::array();
        int32_t n_tokens = 0;
        for (const auto & rank : ranks) {
            results.push_back(json{
                {"index",           json_value(rank, "index", 0)},
                {"relevance_score", json_value(rank, "score", 0.0)},
            });

            n_tokens += json_value(rank, "tokens_evaluated", 0);
        }

        res = json{
            {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
            {"object", "list"},
            {"usage", json{
                {"prompt_tokens", n_tokens},
                {"total_tokens", n_tokens}
            }},
            {"results", results}
        };
    }

    return res;
}

static bool is_valid_utf8(const std::string & str) {
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str.data());
    const unsigned char* end = bytes + str.length();

    while (bytes < end) {
        if (*bytes <= 0x7F) {
            // 1-byte sequence (0xxxxxxx)
            bytes++;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // 2-byte sequence (110xxxxx 10xxxxxx)
            if (end - bytes < 2 || (bytes[1] & 0xC0) != 0x80)
                return false;
            bytes += 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 3 || (bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80)
                return false;
            bytes += 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 4 || (bytes[1] & 0xC0) != 0x80 ||
                (bytes[2] & 0xC0) != 0x80 || (bytes[3] & 0xC0) != 0x80)
                return false;
            bytes += 4;
        } else {
            // Invalid UTF-8 lead byte
            return false;
        }
    }

    return true;
}

static json format_tokenizer_response(const json & tokens) {
    return json {
        {"tokens", tokens}
    };
}

static json format_detokenized_response(const std::string & content) {
    return json {
        {"content", content}
    };
}

static json format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

static std::string safe_json_to_str(const json & data) {
    return data.dump(-1, ' ', false, json::error_handler_t::replace);
}

static std::vector<llama_token_data> get_token_probabilities(llama_context * ctx, int idx) {
    std::vector<llama_token_data> cur;
    const auto * logits = llama_get_logits_ith(ctx, idx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_vocab = llama_vocab_n_tokens(vocab);

    cur.resize(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }

    // sort tokens by logits
    std::sort(cur.begin(), cur.end(), [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    });

    // apply softmax
    float max_l = cur[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < cur.size(); ++i) {
        float p = expf(cur[i].logit - max_l);
        cur[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < cur.size(); ++i) {
        cur[i].p /= cum_sum;
    }

    return cur;
}

static bool are_lora_equal(
        const std::vector<common_adapter_lora_info> & l1,
        const std::vector<common_adapter_lora_info> & l2) {
    if (l1.size() != l2.size()) {
        return false;
    }
    for (size_t i = 0; i < l1.size(); ++i) {
        // we don't check lora.path to reduce the time complexity
        if (l1[i].scale != l2[i].scale || l1[i].ptr != l2[i].ptr) {
            return false;
        }
    }
    return true;
}

// parse lora config from JSON request, returned a copy of lora_base with updated scale
static std::vector<common_adapter_lora_info> parse_lora_request(
        const std::vector<common_adapter_lora_info> & lora_base,
        const json & data) {
    std::vector<common_adapter_lora_info> lora(lora_base);
    int max_idx = lora.size();

    // clear existing value
    for (auto & entry : lora) {
        entry.scale = 0.0f;
    }

    // set value
    for (const auto & entry : data) {
        int id      = json_value(entry, "id", -1);
        float scale = json_value(entry, "scale", 0.0f);
        if (0 <= id && id < max_idx) {
            lora[id].scale = scale;
        } else {
            throw std::runtime_error("invalid adapter id");
        }
    }

    return lora;
}

struct llama_box_params {
    common_params llm_params;

    bool force_context_shift = false; // use context shift even if not allowed
    bool cache_prompt        = true;
    bool endpoint_infill     = false;
    bool endpoint_images     = false;
    int32_t conn_idle        = 60; // connection idle in seconds
    int32_t conn_keepalive   = 15; // connection keep-alive in seconds
    int32_t n_tps            = 0;  // maximum number of tokens per seconds
    int32_t lookup_ngram_min = 0;  // minimum n-gram size for lookup cache
    int32_t max_image_size   = 0;  // maximum image size for vision image processing
};

struct stablediffusion_generated_image {
    int size;
    unsigned char *data;
};

static const std::map<std::string, llm_chat_template> LLM_CHAT_TEMPLATES = {
    { "chatml",            LLM_CHAT_TEMPLATE_CHATML            },
    { "llama2",            LLM_CHAT_TEMPLATE_LLAMA_2           },
    { "llama2-sys",        LLM_CHAT_TEMPLATE_LLAMA_2_SYS       },
    { "llama2-sys-bos",    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS   },
    { "llama2-sys-strip",  LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP },
    { "mistral-v1",        LLM_CHAT_TEMPLATE_MISTRAL_V1        },
    { "mistral-v3",        LLM_CHAT_TEMPLATE_MISTRAL_V3        },
    { "mistral-v3-tekken", LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN },
    { "mistral-v7",        LLM_CHAT_TEMPLATE_MISTRAL_V7        },
    { "phi3",              LLM_CHAT_TEMPLATE_PHI_3             },
    { "phi4",              LLM_CHAT_TEMPLATE_PHI_4             },
    { "falcon3",           LLM_CHAT_TEMPLATE_FALCON_3          },
    { "zephyr",            LLM_CHAT_TEMPLATE_ZEPHYR            },
    { "monarch",           LLM_CHAT_TEMPLATE_MONARCH           },
    { "gemma",             LLM_CHAT_TEMPLATE_GEMMA             },
    { "orion",             LLM_CHAT_TEMPLATE_ORION             },
    { "openchat",          LLM_CHAT_TEMPLATE_OPENCHAT          },
    { "vicuna",            LLM_CHAT_TEMPLATE_VICUNA            },
    { "vicuna-orca",       LLM_CHAT_TEMPLATE_VICUNA_ORCA       },
    { "deepseek",          LLM_CHAT_TEMPLATE_DEEPSEEK          },
    { "deepseek2",         LLM_CHAT_TEMPLATE_DEEPSEEK_2        },
    { "deepseek3",         LLM_CHAT_TEMPLATE_DEEPSEEK_3        },
    { "command-r",         LLM_CHAT_TEMPLATE_COMMAND_R         },
    { "llama3",            LLM_CHAT_TEMPLATE_LLAMA_3           },
    { "chatglm3",          LLM_CHAT_TEMPLATE_CHATGML_3         },
    { "chatglm4",          LLM_CHAT_TEMPLATE_CHATGML_4         },
    { "glmedge",           LLM_CHAT_TEMPLATE_GLMEDGE           },
    { "minicpm",           LLM_CHAT_TEMPLATE_MINICPM           },
    { "exaone3",           LLM_CHAT_TEMPLATE_EXAONE_3          },
    { "rwkv-world",        LLM_CHAT_TEMPLATE_RWKV_WORLD        },
    { "granite",           LLM_CHAT_TEMPLATE_GRANITE           },
    { "gigachat",          LLM_CHAT_TEMPLATE_GIGACHAT          },
    { "megrez",            LLM_CHAT_TEMPLATE_MEGREZ            },
};

const char * llama_chat_template_alias(const char * tmpl) {
    llm_chat_template t = llm_chat_detect_template(std::string(tmpl));
    for (const auto & it : LLM_CHAT_TEMPLATES) {
        if (it.second != t) {
            continue;
        }
        return it.first.c_str();
    }
    return "unknown";
}

typedef minja::chat_template common_chat_template;


bool common_chat_templates_supports_tool_calls(const struct common_chat_templates * tmpls) {
    const auto & tmpl = tmpls->template_tool_use
                            ? *tmpls->template_tool_use
                            : *tmpls->template_default;
    return tmpl.original_caps().supports_tool_calls;
}

  typedef struct llama_chat_function {
        const char * name;
        const char * description;
        const char * parameters;
    } llama_chat_function;

int32_t llm_chat_apply_template2(
    llm_arch arch,
    llm_chat_template tmpl,
    const std::vector<const llama_chat_message *> & chat,
    const std::vector<const llama_chat_function *> & func,
    std::string & dest, bool req_func, bool add_ass) {
    if (tmpl != LLM_CHAT_TEMPLATE_CHATML &&
        tmpl != LLM_CHAT_TEMPLATE_MISTRAL_V7 &&
        tmpl != LLM_CHAT_TEMPLATE_MISTRAL_V3 &&
        tmpl != LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN &&
        tmpl != LLM_CHAT_TEMPLATE_LLAMA_3 &&
        tmpl != LLM_CHAT_TEMPLATE_CHATGML_4 &&
        tmpl != LLM_CHAT_TEMPLATE_GRANITE) {
        return llm_chat_apply_template(tmpl, chat, dest, add_ass);
    }

    // Taken from the research: https://github.com/ggml-org/llama.cpp/issues/5527
    std::stringstream ss;
    if (tmpl == LLM_CHAT_TEMPLATE_CHATML) {
        if (!func.empty()) {
            const llama_chat_message *root_msg = nullptr;
            for (const auto *message: chat) {
                std::string role(message->role);
                if (role == "system") {
                    root_msg = message;
                    break;
                }
            }
            ss << "<|im_start|>system\n";
            if (root_msg) {
                ss << root_msg->content << "\n\n";
            } else {
                ss << "You are a helpful assistant.\n\n";
            }
            if (arch == LLM_ARCH_QWEN2VL) {
                ss << "## Tools\n\n";
            } else {
                ss << "# Tools\n\n";
            }
            if (req_func) {
                ss << "You MUST call one or more functions to assist with the user query. Do not make assumptions about what values to plug into functions.\n\n";
            } else {
                ss << "You CAN call functions to assist with the user query. Do not make assumptions about what values to plug into functions.\n\n";
            }
            if (arch == LLM_ARCH_QWEN2VL) {
                ss << "You are provided with following function tools:\n\n";
                for (const auto & fn : func) {
                    ss << "### " << fn->name << "\n\n";
                    ss << fn->name << ": " << fn->description << " Parameters: " << fn->parameters << "Format the arguments as a JSON object.\n\n";
                }
                if (!req_func) {
                    ss << "When you can reply with your internal knowledge, reply directly without any function calls. ";
                    ss << "Otherwise, try to do function calls without any explanations. ";
                }
                ss << "For each function call, just generate an answer, no explanation before or after your answer, MUST return an JSON object with function name and arguments within <tool_call></tool_call> XML tags:\n";
                ss << "<tool_call>\n";
                ss << "{\"name\": The name of the function to use, \"arguments\": The input of the function, must be an JSON object in compact format}\n";
                ss << "</tool_call>\n";
                ss << "<tool_result>\n";
                ss << "The function results.\n";
                ss << "</tool_result>\n";
                ss << "Reply based on the function results." << "<|im_end|>\n";
            } else {
                ss << "You are provided with following function signatures within <tools></tools> XML tags:\n";
                ss << "<tools>\n";
                for (const auto & fn : func) {
                    ss << R"({"type": "function", "function": {"name": ")" << fn->name << R"(", "description": ")" << fn->description << R"(", "parameters": )" << fn->parameters << "}}\n";
                }
                ss << "</tools>\n\n";
                if (!req_func) {
                    ss << "When you can reply with your internal knowledge, reply directly without any function calls. ";
                    ss << "Otherwise, try to do function calls without any explanations. ";
                }
                ss << "For each function call, just generate an answer, no explanation before or after your answer, MUST return an JSON object with function name and arguments within <tool_call></tool_call> XML tags:\n";
                ss << "<tool_call>\n";
                ss << "{\"name\": <function-name>, \"arguments\": <arguments-json-object>}\n";
                ss << "</tool_call>" << "<|im_end|>\n";
            }
        }
        bool previous_tool_response = false;
        // chatml template
        for (auto message : chat) {
            if (!func.empty()) {
                std::string role(message->role);
                if (role == "tool_call") {
                    if (arch == LLM_ARCH_QWEN2VL) {
                        if (!previous_tool_response) {
                            ss << "<|im_start|>assistant\n";
                        }
                        ss << "<tool_call>\n" << message->content << "\n</tool_call>\n";
                    } else {
                        ss << "<|im_start|>assistant\n";
                        ss << "<tool_call>\n" << message->content << "\n</tool_call>";
                        ss << "<|im_end|>\n";
                    }
                    previous_tool_response = false;
                    continue;
                }
                previous_tool_response = false;
                if (role == "system") {
                    continue;
                }
                if (role == "tool") {
                    if (arch == LLM_ARCH_QWEN2VL) {
                        ss << "<tool_result>\n" << message->content << "\n</tool_result>\n";
                        add_ass = message != chat.back();
                    } else {
                        ss << "<|im_start|>user\n" << message->content << "<|im_end|>\n";
                    }
                    previous_tool_response = true;
                    continue;
                }
                if (role == "assistant" && arch == LLM_ARCH_QWEN2VL) {
                    ss << message->content << "<|im_end|>\n";
                    continue;
                }
            }
            ss << "<|im_start|>" << message->role << "\n" << message->content << "<|im_end|>\n";
        }
        if (add_ass) {
            ss << "<|im_start|>assistant\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V7) {
        // Official mistral 'v7' template
        // See: https://huggingface.co/mistralai/Mistral-Large-Instruct-2411#basic-instruct-template-v7
        // See: https://github.com/mistralai/mistral-common/releases/tag/v1.5.0
        if (!func.empty()) {
            const llama_chat_message *root_msg = nullptr;
            for (const auto *message: chat) {
                std::string role(message->role);
                if (role == "system") {
                    root_msg = message;
                    break;
                }
            }
            ss << "[AVAILABLE_TOOLS] " << "[";
            for (const auto & fn : func) {
                ss << R"({"type": "function", "function": {"name": ")" << fn->name << R"(", "description": ")" << fn->description << R"(", "parameters": )" << fn->parameters << "}}";
                ss << ((fn == func.back()) ? "" : ",");
            }
            ss << "]" << "[/AVAILABLE_TOOLS]";
            if (root_msg) {
                ss << "[SYSTEM_PROMPT] " << root_msg->content;
            } else {
                ss << "[SYSTEM_PROMPT] " << "You are a helpful assistant. ";
            }
            if (req_func) {
                ss << "You MUST call one or more functions to assist with the user query. Do not make assumptions about what values to plug into functions. ";
            } else {
                ss << "You CAN call functions to assist with the user query. Do not make assumptions about what values to plug into functions. ";
                ss << "When you can reply with your internal knowledge, reply directly without any function calls. ";
                ss << "Otherwise, try to call functions without any explanations. ";
            }
            ss << "[/SYSTEM_PROMPT]";
        }
        for (auto message : chat) {
            std::string role(message->role);
            std::string content(message->content);
            if (role == "system") {
                if (!func.empty()) {
                    continue;
                }
                ss << "[SYSTEM_PROMPT] " << content << "[/SYSTEM_PROMPT]";
            } else if (role == "user") {
                ss << "[INST] " << content << "[/INST]";
            }
            else {
                if (!func.empty()) {
                    if (role == "tool_call") {
                        ss << "[TOOL_CALLS] ";
                        ss << "[" << message->content << "]</s>";
                        continue;
                    }
                    if (role == "tool") {
                        ss << "[TOOL_RESULTS] " << message->content << "[/TOOL_RESULTS]";
                        continue;
                    }
                }
                ss << " " << content << "</s>";
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V1
            || tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V3
            || tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN) {
        // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/chat_templates.md
        // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/templates.md
        // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/tool_calling.md
        std::string leading_space = tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V1 ? " " : "";
        std::string trailing_space = tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN ? "" : " ";
        bool trim_assistant_message = tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V3;
        bool is_inside_turn = false;
        if (!func.empty()) {
            const llama_chat_message *root_msg = nullptr;
            for (const auto *message: chat) {
                std::string role(message->role);
                if (role == "system") {
                    root_msg = message;
                    break;
                }
            }
            ss << leading_space << "[AVAILABLE_TOOLS]" << trailing_space << "[";
            for (const auto & fn : func) {
                ss << R"({"type": "function", "function": {"name": ")" << fn->name << R"(", "description": ")" << fn->description << R"(", "parameters": )" << fn->parameters << "}}";
                ss << ((fn == func.back()) ? "" : ",");
            }
            ss << "]" << leading_space << "[/AVAILABLE_TOOLS]";
            ss << leading_space << "[INST]" << trailing_space;
            is_inside_turn = true;
            if (root_msg) {
                ss << root_msg->content << "\n\n";
            } else {
                ss << "You are a helpful assistant.\n\n";
            }
            if (req_func) {
                ss << "You MUST call one or more functions to assist with the user query. Do not make assumptions about what values to plug into functions.\n\n";
            } else {
                ss << "You CAN call functions to assist with the user query. Do not make assumptions about what values to plug into functions. ";
                ss << "When you can reply with your internal knowledge, reply directly without any function calls. ";
                ss << "Otherwise, try to call functions without any explanations.\n\n";
            }
        }
        for (auto message : chat) {
            if (!is_inside_turn) {
                ss << leading_space << "[INST]" << trailing_space;
                is_inside_turn = true;
            }
            std::string role(message->role);
            std::string content(message->content);
            if (role == "system") {
                if (!func.empty()) {
                    continue;
                }
                ss << content << "\n\n";
            } else if (role == "user") {
                ss << content << leading_space << "[/INST]";
            } else {
                if (!func.empty()) {
                    if (role == "tool_call") {
                        ss << leading_space << "[TOOL_CALLS]" << trailing_space;
                        ss << "[" << message->content << "]</s>";
                        continue;
                    }
                    if (role == "tool") {
                        ss << leading_space << "[TOOL_RESULTS]" << trailing_space;
                        ss << message->content;
                        ss << leading_space << "[/TOOL_RESULTS]";
                        continue;
                    }
                }
                ss << trailing_space << (trim_assistant_message ? trim(content) : content) << "</s>";
                is_inside_turn = false;
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_LLAMA_3) {
        if (!func.empty()) {
            const llama_chat_message *root_msg = nullptr;
            for (const auto *message: chat) {
                std::string role(message->role);
                if (role == "system") {
                    root_msg = message;
                    break;
                }
            }
            ss << "<|start_header_id|>system<|end_header_id|>\n\n";
            if (root_msg) {
                ss << trim(root_msg->content) << "\n\n";
            } else {
                ss << "You are a helpful assistant.\n\n";
            }
            if (req_func) {
                ss << "You MUST call one or more functions to assist with the user query. Do not make assumptions about what values to plug into functions.";
            } else {
                ss << "You CAN call functions to assist with the user query. Do not make assumptions about what values to plug into functions.";
            }
            ss << "<|eot_id|>";
        }
        // Llama 3
        for (auto message : chat) {
            std::string role(message->role);
            if (!func.empty()) {
                if (role == "system") {
                    continue;
                }
                if (role == "tool_call") {
                    ss << "<|start_header_id|>assistant<|end_header_id|>\n\n" << trim(message->content) << "<|eot_id|>";
                    continue;
                }
                if (role == "tool") {
                    ss << "<|start_header_id|>ipython<|end_header_id|>\n\n" << trim(message->content) << "<|eot_id|>";
                    continue;
                }
                if (role == "user" && message == chat.back()) {
                    ss << "<|start_header_id|>user<|end_header_id|>\n\n";
                    ss << "You are provided with following function signatures within <tools></tools> XML tags:\n";
                    ss << "<tools>\n";
                    for (const auto & fn : func) {
                        ss << R"({"type": "function", "function": {"name": ")" << fn->name << R"(", "description": ")" << fn->description << R"(", "parameters": )" << fn->parameters << "}}\n";
                    }
                    ss << "</tools>\n\n";
                    if (!req_func) {
                        ss << "When you can reply with your internal knowledge, reply directly without any function call. ";
                        ss << "Otherwise, try to call functions without any explanations. ";
                    }
                    ss << "For each function call, just generate an answer, no explanation before or after your answer, MUST return an JSON object with function name and arguments in the format {\"name\": <function-name>, \"arguments\": <arguments-json-object>}.\n";
                    ss << trim(message->content) << "<|eot_id|>";
                    continue;
                }
            }
            ss << "<|start_header_id|>" << role << "<|end_header_id|>\n\n" << trim(message->content) << "<|eot_id|>";
        }
        if (add_ass) {
            ss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_CHATGML_4) {
        ss << "[gMASK]" << "<sop>";
        if (!func.empty()) {
            const llama_chat_message *root_msg = nullptr;
            for (const auto *message: chat) {
                std::string role(message->role);
                if (role == "system") {
                    root_msg = message;
                    break;
                }
            }
            ss << "<|system|>\n";
            if (root_msg) {
                ss << root_msg -> content << "\n";
            } else {
                ss << "You are a helpful assistant.\n";
            }
            if (req_func) {
                ss << "You MUST call one or more functions to assist with the user query. Do not make assumptions about what values to plug into functions.\n";
            } else {
                ss << "You CAN call functions to assist with the user query. Do not make assumptions about what values to plug into functions.\n";
            }
            ss << "# Functions\n";
            ss << "You are provided with following functions:\n";
            for (size_t i = 0; i < func.size(); i++) {
                const llama_chat_function *fn = func[i];
                ss << "## Function " << i << "\n";
                ss << "### Name\n" << fn->name << "\n";
                ss << "### Description\n" << fn->description << "\n";
                ss << "### Parameters\n" << fn->parameters << "\n";
            }
            if (!req_func) {
                ss << "When you can reply with your internal knowledge, reply directly without any function calls. ";
                ss << "Otherwise, try to call functions without any explanations. ";
            }
            ss << "For each function call, just generate an answer, no explanation before or after your answer, MUST return an JSON object with function name and arguments within <tool_call></tool_call> XML tags:\n";
            ss << "<tool_call>\n";
            ss << "{\"name\": The name of the function to use, \"arguments\": The input of the function, must be an JSON object in compact format}\n";
            ss << "</tool_call>\n";
            ss << "<tool_result>\n";
            ss << "The function results.\n";
            ss << "</tool_result>\n";
            ss << "Reply based on the function results.\n";
        }
        bool previous_tool_response = false;
        for (auto message : chat) {
            std::string role(message->role);
            if (!func.empty()) {
                if (role == "tool_call") {
                    if (!previous_tool_response) {
                        ss << "<|assistant|>\n";
                    }
                    ss << "<tool_call>\n" << message->content << "\n</tool_call>\n";
                    previous_tool_response = false;
                    continue;
                }
                previous_tool_response = false;
                if (role == "system") {
                    continue;
                }
                if (role == "tool") {
                    ss << "<tool_result>\n" << message->content << "\n</tool_result>\n";
                    add_ass = message != chat.back();
                    previous_tool_response = true;
                    continue;
                }
                if (role == "assistant") {
                    ss << "<|assistant|>\n" << message->content;
                    continue;
                }
            }
            ss << "<|" << role << "|>" << "\n" << message->content;
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_GRANITE) {
        if (!func.empty()) {
            const llama_chat_message *root_msg = nullptr;
            for (const auto *message: chat) {
                std::string role(message->role);
                if (role == "system") {
                    root_msg = message;
                    break;
                }
            }
            ss << "<|start_of_role|>tools<|end_of_role|>[";
            for (const auto & fn : func) {
                ss << R"({"type": "function", "function": {"name": ")" << fn->name << R"(", "description": ")" << fn->description << R"(", "parameters": )" << fn->parameters << "}}";
                ss << ((fn == func.back()) ? "" : ",");
            }
            ss << "]<|end_of_text|>\n";
            ss << "<|start_of_role|>system<|end_of_role|>";
            if (root_msg) {
                ss << trim(root_msg->content) << " ";
            } else {
                ss << "You are a helpful assistant with tool calling capabilities. ";
            }
            if (req_func) {
                ss << "You MUST call one or more tools to assist with the user query. Do not make assumptions about what values to plug into tools. ";
            } else {
                ss << "You CAN call tools to assist with the user query. Do not make assumptions about what values to plug into tools. ";
            }
            if (!req_func) {
                ss << "When you can reply with your internal knowledge, reply directly without any tool calls. ";
                ss << "Otherwise, try to call tools without any explanations. ";
            }
            ss << "For each tool call, just generate an answer, no explanation before or after your answer, MUST return <|tool_call|><tool_call> followed by an JSON list of tool used as follows: ";
            ss << R"(<|tool_call|><tool_call>[{"name": <function-name>, "arguments": <arguments-json-object>}])";
            ss << "Write the response to the user's input by strictly aligning with the facts in the provided documents.";
            ss << "<|end_of_text|>\n";
        }
        // IBM Granite template
        for (const auto & message : chat) {
            std::string role(message->role);
            if (!func.empty()) {
                if (role == "system") {
                    continue;
                }
                if (role == "tool_call") {
                    ss << "<|start_of_role|>assistant<|start_of_role|><|tool_call|><tool_call>" << message->content << "<|end_of_text|>\n";
                    continue;
                }
                if (role == "tool") {
                    ss << "<|start_of_role|>tool_response<|end_of_role|>" << message->content << "<|end_of_text|>\n";
                    continue;
                }
            }
            ss << "<|start_of_role|>" << role << "<|end_of_role|>";
            ss << message->content << "<|end_of_text|>\n";
        }
        if (add_ass) {
            ss << "<|start_of_role|>assistant<|end_of_role|>\n";
        }
    } else {
        // template not supported
        return -1;
    }
    dest = ss.str();
    return dest.size();
}
    
int32_t llama_chat_apply_template2(
                const struct llama_model * model,
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
        const struct llama_chat_function * func,
                                    size_t n_func,
                                      bool req_func,
                                      bool add_ass,
                                    char * buf,
                                 int32_t   length) {
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl);

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    // format the func to string
    std::vector<const llama_chat_function *> func_vec;
    func_vec.resize(n_func);
    for (size_t i = 0; i < n_func; i++) {
        func_vec[i] = &func[i];
    }

    std::string       formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llm_chat_apply_template2(model ? model->arch : LLM_ARCH_LLAMA, detected_tmpl, chat_vec, func_vec,
                                          formatted_chat, req_func, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}


static common_chat_params common_chat_templates_apply_legacy2(
   const struct llama_model * model,
    const struct common_chat_templates * tmpls,
    const struct common_chat_templates_inputs & inputs)
{
    int alloc_size = 0;
    std::vector<llama_chat_message> chat;
    std::vector<std::string> roles;
    std::vector<std::string> contents;
    for (const common_chat_msg & msg : inputs.messages) {
        std::string role = msg.role;
        std::string content = msg.content;
        if (msg.tool_calls.empty()) {
            for (const common_chat_msg_content_part & part : msg.content_parts) {
                if (part.type != "text") {
                    LOG_WRN("Ignoring non-text content part: %s\n", part.type.c_str());
                    continue;
                }
                if (!content.empty()) {
                    content += "\n";;
                }
                content += part.text;
            }
        } else {
            role = "tool_call";
            for (const common_chat_tool_call & tc : msg.tool_calls) {
                if (!content.empty()) {
                    content += "\n";
                }
                content += "{\"name\":\"" + tc.name + "\",\"arguments\":" + tc.arguments + "}";
            }
        }
        roles.emplace_back(role);
        contents.emplace_back(content);
    }
    for (size_t i = 0; i < contents.size(); ++i) {
        const std::string & role = roles[i];
        const std::string & content = contents[i];
        chat.push_back({role.c_str(), content.c_str()});
        alloc_size += (role.size() + content.size()) * 1.25;
    }
    std::vector<llama_chat_function> func;
    for (const common_chat_tool & tool : inputs.tools) {
        func.push_back({tool.name.c_str(), tool.description.c_str(), tool.parameters.c_str()});
        alloc_size += (tool.name.size() + tool.description.size() + tool.parameters.size()) * 1.25;
    }

    std::vector<char> buf(alloc_size);

    // run the first time to get the total output length
    const auto & src = tmpls->template_default->source();
    int32_t res = llama_chat_apply_template2(model, src.c_str(), chat.data(), chat.size(), func.data(), func.size(), inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, inputs.add_generation_prompt, buf.data(), buf.size());

    // error: chat template is not supported
    if (res < 0) {
        // if the custom "tmpl" is not supported, we throw an error
        // this is a bit redundant (for good), since we're not sure if user validated the custom template with llama_chat_verify_template()
        throw std::runtime_error("this custom template is not supported");
    }

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template2(model, src.c_str(), chat.data(), chat.size(), func.data(), func.size(), inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, inputs.add_generation_prompt, buf.data(), buf.size());
    }

    common_chat_params params;
    params.prompt = std::string(buf.data(), res);
    if (!inputs.json_schema.empty()) {
        params.grammar = json_schema_to_grammar(json::parse(inputs.json_schema));
    } else {
        params.grammar = inputs.grammar;
    }
    return params;
}


common_chat_params common_chat_templates_apply2(
    const struct llama_model * model,
    const struct common_chat_templates * tmpls,
    const struct common_chat_templates_inputs & inputs)
{
    GGML_ASSERT(tmpls != nullptr);
    return inputs.use_jinja
               ? common_chat_templates_apply_jinja(tmpls, inputs)
               : common_chat_templates_apply_legacy2(model, tmpls, inputs);
}

struct llava_text_token_batch_wrapper {
    std::vector<llama_pos> pos;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t> logits;
    llama_batch batch;

    llava_text_token_batch_wrapper(llama_token *token, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
        pos.resize(n_tokens);
        n_seq_id.resize(n_tokens);
        seq_ids.resize(n_tokens + 1);
        logits.resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0]       = seq_id;
        seq_ids[n_tokens] = nullptr;
        batch             = {
            /*n_tokens       =*/n_tokens,
            /*tokens         =*/token,
            /*embd           =*/nullptr,
            /*pos            =*/pos.data(),
            /*n_seq_id       =*/n_seq_id.data(),
            /*seq_id         =*/seq_ids.data(),
            /*logits         =*/logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.pos[i]      = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i]   = seq_id_0.data();
            batch.logits[i]   = false;
        }
    }
};

struct llava_image_embed_batch_wrapper {
    std::vector<llama_pos> pos;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t> logits;
    llama_batch batch;

    llava_image_embed_batch_wrapper(float *embd, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
        pos.resize(n_tokens);
        n_seq_id.resize(n_tokens);
        seq_ids.resize(n_tokens + 1);
        logits.resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0]       = seq_id;
        seq_ids[n_tokens] = nullptr;
        batch             = {
            /*n_tokens       =*/n_tokens,
            /*tokens         =*/nullptr,
            /*embd           =*/embd,
            /*pos            =*/pos.data(),
            /*n_seq_id       =*/n_seq_id.data(),
            /*seq_id         =*/seq_ids.data(),
            /*logits         =*/logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.pos[i]      = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i]   = seq_id_0.data();
            batch.logits[i]   = false;
        }
    }
};

struct qwen2vl_text_token_batch_wrapper {
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t> logits;
    llama_batch batch;

    qwen2vl_text_token_batch_wrapper(llama_token *token, int32_t n_tokens, llama_pos *pos, llama_seq_id seq_id) {
        n_seq_id.resize(n_tokens);
        seq_ids.resize(n_tokens + 1);
        logits.resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0]       = seq_id;
        seq_ids[n_tokens] = nullptr;
        batch             = {
            /*n_tokens       =*/n_tokens,
            /*tokens         =*/token,
            /*embd           =*/nullptr,
            /*pos            =*/pos,
            /*n_seq_id       =*/n_seq_id.data(),
            /*seq_id         =*/seq_ids.data(),
            /*logits         =*/logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id[i]   = seq_id_0.data();
            batch.logits[i]   = false;
        }
    }
};

struct qwen2vl_image_embed_batch_wrapper {
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t> logits;
    llama_batch batch;

    qwen2vl_image_embed_batch_wrapper(float *embd, int32_t n_tokens, llama_pos *pos, llama_seq_id seq_id) {
        n_seq_id.resize(n_tokens);
        seq_ids.resize(n_tokens + 1);
        logits.resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0]       = seq_id;
        seq_ids[n_tokens] = nullptr;
        batch             = {
            /*n_tokens       =*/n_tokens,
            /*tokens         =*/nullptr,
            /*embd           =*/embd,
            /*pos            =*/pos,
            /*n_seq_id       =*/n_seq_id.data(),
            /*seq_id         =*/seq_ids.data(),
            /*logits         =*/logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id[i]   = seq_id_0.data();
            batch.logits[i]   = false;
        }
    }
};



void common_batch_add_with_mrope(
    struct llama_batch &batch,
    llama_token id,
    llama_pos st_pos_id,
    int32_t n_eval,
    const std::vector<llama_seq_id> &seq_ids,
    bool logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token[batch.n_tokens] = id;
    for (int i = 0; i < 4; i++) {
        if (i == 3) {
            st_pos_id = 0;
        }
        batch.pos[batch.n_tokens + n_eval * i] = st_pos_id;
    }
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t j = 0; j < seq_ids.size(); ++j) {
        batch.seq_id[batch.n_tokens][j] = seq_ids[j];
    }
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

bool clip_is_gemma3(const struct clip_ctx * ctx) {
    return ctx->proj_type == PROJECTOR_TYPE_GEMMA3;
}

//typedef enum
//{
//    STBIR_TYPE_UINT8 ,
//    STBIR_TYPE_UINT16,
//    STBIR_TYPE_UINT32,
//    STBIR_TYPE_FLOAT ,
//
//    STBIR_MAX_TYPES
//} stbir_datatype;

const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
};

static std::string get_all_cache_kv_types_string() {
    std::ostringstream msg;
    for (const auto &type : kv_cache_types) {
        msg << ggml_type_name(type) << (&type == &kv_cache_types.back() ? "" : ", ");
    }
    return msg.str();
}

static ggml_type parse_cache_kv_type(const std::string &s) {
    for (const auto &type : kv_cache_types) {
        if (ggml_type_name(type) == s) {
            return type;
        }
    }
    throw std::runtime_error("Unsupported cache type: " + s);
}

static std::vector<const char *> get_builtin_chat_templates() {
    std::vector<const char *> tmpls;
    int32_t res = llama_chat_builtin_templates(nullptr, 0);
    tmpls.resize(res);
    llama_chat_builtin_templates(tmpls.data(), tmpls.size());
    return tmpls;
}

static std::string get_builtin_chat_templates_string() {
    std::vector<const char *> tmpls = get_builtin_chat_templates();
    std::ostringstream msg;
    for (const auto &tmpl : tmpls) {
        msg << tmpl << (&tmpl == &tmpls.back() ? "" : ", ");
    }
    return msg.str();
}

inline std::vector<ggml_backend_dev_t> parse_device_list(const std::string &value) {
    std::vector<ggml_backend_dev_t> devices;
    auto dev_names = string_split<std::string>(value, ',');
    if (dev_names.empty()) {
        throw std::invalid_argument("no devices specified");
    }
    if (dev_names.size() == 1 && dev_names[0] == "none") {
        devices.push_back(nullptr);
    } else {
        for (const auto &device : dev_names) {
            auto *dev = ggml_backend_dev_by_name(device.c_str());
            if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                throw std::invalid_argument(string_format("invalid device: %s", device.c_str()));
            }
            devices.push_back(dev);
        }
        devices.push_back(nullptr);
    }
    return devices;
}


//
// Environment variable utils
//

template <typename T>
static typename std::enable_if<std::is_same<T, std::string>::value, void>::type get_env(std::string name, T &target) {
    char *value = std::getenv(name.c_str());
    target      = value ? std::string(value) : target;
}

template <typename T>
static typename std::enable_if<!std::is_same<T, bool>::value && std::is_integral<T>::value, void>::type get_env(std::string name, T &target) {
    char *value = std::getenv(name.c_str());
    target      = value ? std::stoi(value) : target;
}

template <typename T>
static typename std::enable_if<std::is_floating_point<T>::value, void>::type get_env(std::string name, T &target) {
    char *value = std::getenv(name.c_str());
    target      = value ? std::stof(value) : target;
}

template <typename T>
static typename std::enable_if<std::is_same<T, bool>::value, void>::type get_env(std::string name, T &target) {
    char *value = std::getenv(name.c_str());
    if (value) {
        std::string val(value);
        target = val == "1" || val == "true";
    }
}

template <typename T>
static typename std::enable_if<std::is_same<T, std::vector<ggml_backend_dev_t>>::value, void>::type get_env(std::string name, T &target) {
    char *value = std::getenv(name.c_str());
    if (value) {
        target = parse_device_list(value);
    }
}

static void unknown(const char *flag) {
    fprintf(stderr, "Unknown argument: %s\n", flag);
}

[[noreturn]] static void missing(const char *flag) {
    throw std::invalid_argument("Missing argument: " + std::string(flag));
}

[[noreturn]] static void invalid(const char *flag) {
    throw std::invalid_argument("Invalid argument: " + std::string(flag));
}



static void llama_box_params_print_usage(int, char **argv, const llama_box_params &params_) {
    struct opt {
        LLAMA_COMMON_ATTRIBUTE_FORMAT(4, 5)

        opt(std::string tags, const char *args, const char *desc, ...)
            : tags(std::move(tags)), args(args), desc(desc) {
            va_list args_list;
            va_start(args_list, desc);
            char buffer[1024];
            vsnprintf(buffer, sizeof(buffer), desc, args_list);
            va_end(args_list);
            this->desc = buffer;
        }

        opt(std::string grp)
            : grp(std::move(grp)) {
        }

        std::string tags;
        std::string args;
        std::string desc;
        std::string grp;
    };

    const auto &llm_params = params_.llm_params;

    std::string default_sampler_type_chars;
    std::string default_sampler_type_names;
    for (const auto &sampler : llm_params.sampling.samplers) {
        default_sampler_type_chars += common_sampler_type_to_chr(sampler);
        default_sampler_type_names += common_sampler_type_to_str(sampler);
        default_sampler_type_names += (&sampler == &llm_params.sampling.samplers.back() ? "" : ";");
    }

    std::string default_dry_sequence_breaker_names;
    for (const auto &breaker : llm_params.sampling.dry_sequence_breakers) {
        default_dry_sequence_breaker_names += breaker;
        default_dry_sequence_breaker_names += (&breaker == &llm_params.sampling.dry_sequence_breakers.back() ? "" : ";");
    }

    // clang-format off
    std::vector<opt> opts;
    // general //
    opts.push_back({ "general" });
    opts.push_back({ "general",                            "-h,    --help, --usage",                        "Print usage and exit" });
    opts.push_back({ "general",                            "       --version",                              "Print version and exit" });
    opts.push_back({ "general",                            "       --system-info",                          "Print system info and exit" });
    opts.push_back({ "general",                            "       --list-devices",                         "Print list of available devices and exit" });
    opts.push_back({ "general",                            "-v,    --verbose, --log-verbose",               "Set verbosity level to infinity (i.e. log all messages, useful for debugging)" });
    opts.push_back({ "general",                            "-lv,   --verbosity, --log-verbosity V",         "Set the verbosity threshold, messages with a higher verbosity will be ignored" });
    opts.push_back({ "general",                            "       --log-colors",                           "Enable colored logging" });
    // general //
    // server //
    opts.push_back({ "server" });
    opts.push_back({ "server",                             "       --host HOST",                            "IP address to listen (default: %s)", llm_params.hostname.c_str() });
    opts.push_back({ "server",                             "       --port PORT",                            "Port to listen (default: %d)", llm_params.port });
    opts.push_back({ "server",                             "-to    --timeout N",                            "Server read/write timeout in seconds (default: %d)", llm_params.timeout_read });
    opts.push_back({ "server",                             "       --threads-http N",                       "Number of threads used to process HTTP requests (default: %d)", llm_params.n_threads_http });
    opts.push_back({ "server",                             "       --conn-idle N",                          "Server connection idle in seconds (default: %d)", params_.conn_idle });
    opts.push_back({ "server",                             "       --conn-keepalive N",                     "Server connection keep-alive in seconds (default: %d)", params_.conn_keepalive });
    opts.push_back({ "server",                             "-m,    --model FILE",                           "Model path (default: %s)", DEFAULT_MODEL_PATH });
    opts.push_back({ "server",                             "-a,    --alias NAME",                           "Model name alias" });
    opts.push_back({ "server",                             "       --lora FILE",                            "Apply LoRA adapter (implies --no-mmap)" });
    opts.push_back({ "server",                             "       --lora-scaled FILE SCALE",               "Apply LoRA adapter with user defined scaling S (implies --no-mmap)" });
    opts.push_back({ "server",                             "       --lora-init-without-apply",              "Load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: %s)", llm_params.lora_init_without_apply ? "enabled" : "disabled" });
    opts.push_back({ "server",                             "-s,    --seed N",                               "RNG seed (default: %d, use random seed for %d)", llm_params.sampling.seed, LLAMA_DEFAULT_SEED });
    opts.push_back({ "server",                             "       --no-flash-attn",                        "Disable Flash Attention, which can increase (V)RAM but reduce computation" });
    opts.push_back({ "server",                             "-fa,   --flash-attn",                           "Enable Flash Attention, which can reduce (V)RAM but increase computation" });
    opts.push_back({ "server",                             "       --metrics",                              "Enable prometheus compatible metrics endpoint (default: %s)", llm_params.endpoint_metrics ? "enabled" : "disabled" });
    opts.push_back({ "server",                             "       --infill",                               "Enable infill endpoint (default: %s)", params_.endpoint_infill? "enabled" : "disabled" });
    opts.push_back({ "server",                             "       --embeddings",                           "Enable embedding endpoint (default: %s)", llm_params.embedding ? "enabled" : "disabled" });
    opts.push_back({ "server",                             "       --images",                               "Enable image endpoint (default: %s)", params_.endpoint_images ? "enabled" : "disabled" });
    opts.push_back({ "server",                             "       --rerank",                               "Enable reranking endpoint (default: %s)", llm_params.reranking ? "enabled" : "disabled" });
    opts.push_back({ "server",                             "       --slots",                                "Enable slots monitoring endpoint (default: %s)", llm_params.endpoint_slots ? "enabled" : "disabled" });
    opts.push_back({ "server",                             "       --rpc SERVERS",                          "A comma-separated list of RPC server" });
    opts.push_back({ "server",                             "-ts,   --tensor-split SPLIT",                   "Fraction of the model to offload to each device, comma-separated list of proportions, e.g. 3,1\n"
                                                                                                            "For image models, indicate which device should be able to offload"});
    opts.push_back({ "server",                             "-ngl,  --gpu-layers,  --n-gpu-layers N",        "Number of layers to store in VRAM\n"
                                                                                                            "-ngl 0 means no offloading"});
    opts.push_back({ "server",                             "       --no-warmup",                            "Disable warm up the model with an empty run" });
    opts.push_back({ "server",                             "       --warmup",                               "Enable warm up the model with an empty run, which is used to occupy the (V)RAM before serving" });
    // server // completion //
    opts.push_back({ "server/completion" });
    opts.push_back({ "server/completion",                  "-dev,  --device <dev1,dev2,...>",               "A comma-separated list of devices to use for offloading (none = don't offload)\n"
                                                                                                            "Use --list-devices to see a list of available devices"});
    opts.push_back({ "server/completion",                  "-sm,   --split-mode SPLIT_MODE",                "How to split the model across multiple GPUs, one of:\n"
                                                                                                            "  - none: use one GPU only\n"
                                                                                                            "  - layer (default): split layers and KV across GPUs\n"
                                                                                                            "  - row: split rows across GPUs, store intermediate results and KV in --main-gpu" });
    opts.push_back({ "server/completion",                  "-mg,   --main-gpu N",                           "The device to use for the model\n"
                                                                                                            "Work with --split-mode none|row, or indicate the device to offload projector model specified by --mmproj (default: %d)", llm_params.main_gpu });
    opts.push_back({ "server/completion",                  "       --override-kv KEY=TYPE:VALUE",           "Advanced option to override model metadata by key, may be specified multiple times\n"
                                                                                                            "Types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false" });
    opts.push_back({ "server/completion",                  "       --chat-template BUILTIN",                "Set built-in chat template (default: analyze from model's metadata)\n"
                                                                                                            "Only built-in templates are accepted, implicit reset --jinja setting\n"
                                                                                                            "List of built-in templates: %s", get_builtin_chat_templates_string().c_str() });
    opts.push_back({ "server/completion",                  "       --jinja",                                "Enable jinja template for chat, implicit reset --chat-template and --chat-template-file setting (default: disabled)" });
    opts.push_back({ "server/completion",                  "       --chat-template-file FILE",              "Set jinja chat template (default: take from model's metadata)\n"
                                                                                                            "Required --jinja set before\n" });
    opts.push_back({ "server/completion",                  "       --slot-save-path PATH",                  "Path to save slot kv cache (default: disabled)" });
    opts.push_back({ "server/completion",                  "-sps,  --slot-prompt-similarity N",             "How much the prompt of a request must match the prompt of a slot in order to use that slot (default: %.2f, 0.0 = disabled)\n", llm_params.slot_prompt_similarity });
    opts.push_back({ "server/completion",                  "-tps   --tokens-per-second N",                  "Maximum number of tokens per second (default: %d, 0 = disabled, -1 = try to detect)\n"
                                                                                                            "When enabled, limit the request within its X-Request-Tokens-Per-Second HTTP header", params_.n_tps });
    opts.push_back({ "server/completion",                  "-t,    --threads N",                            "Number of threads to use during generation (default: %d)", llm_params.cpuparams.n_threads });
#ifndef GGML_USE_OPENMP
    opts.push_back({ "server/completion",                  "-C,    --cpu-mask M",                           "Set CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")"});
    opts.push_back({ "server/completion",                  "-Cr,   --cpu-range lo-hi",                      "Range of CPUs for affinity. Complements --cpu-mask"});
    opts.push_back({ "server/completion",                  "       --cpu-strict <0|1>",                     "Use strict CPU placement (default: %u)\n", (unsigned) llm_params.cpuparams.strict_cpu});
    opts.push_back({ "server/completion",                  "       --prio N",                               "Set process/thread priority (default: %d), one of:\n"
                                                                                                            "  - 0-normal\n"
                                                                                                            "  - 1-medium\n"
                                                                                                            "  - 2-high\n"
                                                                                                            "  - 3-realtime", llm_params.cpuparams.priority});
    opts.push_back({ "server/completion",                  "       --poll <0...100>",                       "Use polling level to wait for work (0 - no polling, default: %u)\n", (unsigned) llm_params.cpuparams.poll});
#endif
    opts.push_back({ "server/completion",                  "-tb,   --threads-batch N",                      "Number of threads to use during batch and prompt processing (default: same as --threads)" });
#ifndef GGML_USE_OPENMP
    opts.push_back({ "server/completion",                  "-Cb,   --cpu-mask-batch M",                     "Set CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)"});
    opts.push_back({ "server/completion",                  "-Crb,  --cpu-range-batch lo-hi",                "Ranges of CPUs for affinity. Complements --cpu-mask-batch"});
    opts.push_back({ "server/completion",                  "       --cpu-strict-batch <0|1>",               "Use strict CPU placement (default: same as --cpu-strict)"});
    opts.push_back({ "server/completion",                  "       --prio-batch N",                         "Set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: --priority)"});
    opts.push_back({ "server/completion",                  "       --poll-batch <0...100>",                 "Use polling to wait for work (default: same as --poll"});
#endif
    opts.push_back({ "server/completion",                  "-c,    --ctx-size N",                           "Size of the prompt context (default: %d, 0 = loaded from model)", llm_params.n_ctx });
    opts.push_back({ "server/completion",                  "       --no-context-shift",                     "Disable context shift on infinite text generation and long prompt embedding" });
    opts.push_back({ "server/completion",                  "       --context-shift",                        "Enable context shift on infinite text generation and long prompt embedding" });
    opts.push_back({ "server/completion",                  "-n,    --predict N",                            "Number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)", llm_params.n_predict });
    opts.push_back({ "server/completion",                  "-b,    --batch-size N",                         "Logical batch size.\n"
                                                                                                            "Increasing this value above the value of the physical batch size may improve prompt processing performance when using multiple GPUs with pipeline parallelism. (default: %d)", llm_params.n_batch });
    opts.push_back({ "server/completion",                  "-ub,   --ubatch-size N",                        "Physical batch size, which is the maximum number of tokens that may be processed at a time.\n"
                                                                                                            "Increasing this value may improve performance during prompt processing, at the expense of higher memory usage. (default: %d)", llm_params.n_ubatch });
    opts.push_back({ "server/completion",                  "       --keep N",                               "Number of tokens to keep from the initial prompt (default: %d, -1 = all)", llm_params.n_keep });
    opts.push_back({ "server/completion",                  "       --no-escape",                            "Disable process escape sequences" });
    opts.push_back({ "server/completion",                  "-e,    --escape",                               R"(Process escapes sequences (\n, \r, \t, \', \", \\) (default: %s))", llm_params.escape ? "true" : "false" });
    opts.push_back({ "server/completion",                  "       --samplers SAMPLERS",                    "Samplers that will be used for generation in the order, separated by ';' (default: %s)", default_sampler_type_names.c_str() });
    opts.push_back({ "server/completion",                  "       --sampling-seq SEQUENCE",                "Simplified sequence for samplers that will be used (default: %s)", default_sampler_type_chars.c_str() });
    opts.push_back({ "server/completion",                  "       --temp T",                               "Temperature (default: %.1f)", (double)llm_params.sampling.temp });
    opts.push_back({ "server/completion",                  "       --top-k N",                              "Top-K sampling (default: %d, 0 = disabled)", llm_params.sampling.top_k });
    opts.push_back({ "server/completion",                  "       --top-p N",                              "Top-P sampling (default: %.1f, 1.0 = disabled)", (double) llm_params.sampling.top_p });
    opts.push_back({ "server/completion",                  "       --min-p N",                              "Min-P sampling (default: %.1f, 0.0 = disabled)", (double)llm_params.sampling.min_p });
    opts.push_back({ "server/completion",                  "       --top-nsigma N",                         "Top-N-Sigma sampling (default: %.1f, -1.0 = disabled)", (double)llm_params.sampling.top_n_sigma });
    opts.push_back({ "server/completion",                  "       --xtc-probability N",                    "XTC probability (default: %.1f, 0.0 = disabled)", (double)llm_params.sampling.xtc_probability });
    opts.push_back({ "server/completion",                  "       --xtc-threshold N",                      "XTC threshold (default: %.1f, 1.0 = disabled)", (double)llm_params.sampling.xtc_threshold });
    opts.push_back({ "server/completion",                  "       --typical N",                            "Locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)", (double)llm_params.sampling.typ_p });
    opts.push_back({ "server/completion",                  "       --repeat-last-n N",                      "Last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)", llm_params.sampling.penalty_last_n });
    opts.push_back({ "server/completion",                  "       --repeat-penalty N",                     "Penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)", (double)llm_params.sampling.penalty_repeat });
    opts.push_back({ "server/completion",                  "       --presence-penalty N",                   "Repeat alpha presence penalty (default: %.1f, 0.0 = disabled)", (double)llm_params.sampling.penalty_present });
    opts.push_back({ "server/completion",                  "       --frequency-penalty N",                  "Repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)", (double)llm_params.sampling.penalty_freq });
    opts.push_back({ "server/completion",                  "       --dry-multiplier N",                     "Set DRY sampling multiplier (default: %.1f, 0.0 = disabled)", (double)llm_params.sampling.dry_multiplier });
    opts.push_back({ "server/completion",                  "       --dry-base N",                           "Set DRY sampling base value (default: %.2f)", (double)llm_params.sampling.dry_base });
    opts.push_back({ "server/completion",                  "       --dry-allowed-length N",                 "Set allowed length for DRY sampling (default: %d)", llm_params.sampling.dry_allowed_length });
    opts.push_back({ "server/completion",                  "       --dry-penalty-last-n N",                 "Set DRY penalty for the last n tokens (default: %d, 0 = disable, -1 = context size)", llm_params.sampling.dry_penalty_last_n });
    opts.push_back({ "server/completion",                  "       --dry-sequence-breaker N",               "Add sequence breaker for DRY sampling, clearing out default breakers (%s) in the process; use \"none\" to not use any sequence breakers", default_dry_sequence_breaker_names.c_str() });
    opts.push_back({ "server/completion",                  "       --dynatemp-range N",                     "Dynamic temperature range (default: %.1f, 0.0 = disabled)", (double)llm_params.sampling.dynatemp_range });
    opts.push_back({ "server/completion",                  "       --dynatemp-exp N",                       "Dynamic temperature exponent (default: %.1f)", (double)llm_params.sampling.dynatemp_exponent });
    opts.push_back({ "server/completion",                  "       --mirostat N",                           "Use Mirostat sampling, Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used (default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", llm_params.sampling.mirostat });
    opts.push_back({ "server/completion",                  "       --mirostat-lr N",                        "Mirostat learning rate, parameter eta (default: %.1f)", (double)llm_params.sampling.mirostat_eta });
    opts.push_back({ "server/completion",                  "       --mirostat-ent N",                       "Mirostat target entropy, parameter tau (default: %.1f)", (double)llm_params.sampling.mirostat_tau });
    opts.push_back({ "server/completion",                  "-l     --logit-bias TOKEN_ID(+/-)BIAS",         R"(Modifies the likelihood of token appearing in the completion, i.e. "--logit-bias 15043+1" to increase likelihood of token ' Hello', or "--logit-bias 15043-1" to decrease likelihood of token ' Hello')" });
    opts.push_back({ "server/completion",                  "       --grammar GRAMMAR",                      "BNF-like grammar to constrain generations (see samples in grammars/ dir) (default: '%s')", llm_params.sampling.grammar.c_str() });
    opts.push_back({ "server/completion",                  "       --grammar-file FILE",                    "File to read grammar from" });
    opts.push_back({ "server/completion",                  "-j,    --json-schema SCHEMA",                   "JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object. For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead" });
    opts.push_back({ "server/completion",                  "       --rope-scaling {none,linear,yarn}",      "RoPE frequency scaling method, defaults to linear unless specified by the model" });
    opts.push_back({ "server/completion",                  "       --rope-scale N",                         "RoPE context scaling factor, expands context by a factor of N" });
    opts.push_back({ "server/completion",                  "       --rope-freq-base N",                     "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)" });
    opts.push_back({ "server/completion",                  "       --rope-freq-scale N",                    "RoPE frequency scaling factor, expands context by a factor of 1/N" });
    opts.push_back({ "server/completion",                  "       --yarn-orig-ctx N",                      "YaRN original context size of model (default: %d = model training context size)", llm_params.yarn_orig_ctx });
    opts.push_back({ "server/completion",                  "       --yarn-ext-factor N",                    "YaRN extrapolation mix factor (default: %.1f, 0.0 = full interpolation)", (double)llm_params.yarn_ext_factor });
    opts.push_back({ "server/completion",                  "       --yarn-attn-factor N",                   "YaRN scale sqrt(t) or attention magnitude (default: %.1f)", (double)llm_params.yarn_attn_factor });
    opts.push_back({ "server/completion",                  "       --yarn-beta-fast N",                     "YaRN low correction dim or beta (default: %.1f)", (double)llm_params.yarn_beta_fast });
    opts.push_back({ "server/completion",                  "       --yarn-beta-slow N",                     "YaRN high correction dim or alpha (default: %.1f)", (double)llm_params.yarn_beta_slow });
    opts.push_back({ "server/completion",                  "-nkvo, --no-kv-offload",                        "Disable KV offload" });
    opts.push_back({ "server/completion",                  "       --no-cache-prompt",                      "Disable caching prompt" });
    opts.push_back({ "server/completion",                  "       --cache-reuse N",                        "Min chunk size to attempt reusing from the cache via KV shifting (default: %d)", llm_params.n_cache_reuse });
    opts.push_back({ "server/completion",                  "-ctk,  --cache-type-k TYPE",                    "KV cache data type for K, allowed values: %s (default: %s)", get_all_cache_kv_types_string().c_str(), ggml_type_name(llm_params.cache_type_k) });
    opts.push_back({ "server/completion",                  "-ctv,  --cache-type-v TYPE",                    "KV cache data type for V, allowed values: %s (default: %s)", get_all_cache_kv_types_string().c_str(), ggml_type_name(llm_params.cache_type_v) });
    opts.push_back({ "server/completion",                  "-dt,   --defrag-thold N",                       "KV cache defragmentation threshold (default: %.1f, < 0 - disabled)", (double)llm_params.defrag_thold });
    opts.push_back({ "server/completion",                  "-np,   --parallel N",                           "Number of parallel sequences to decode (default: %d)", llm_params.n_parallel });
    opts.push_back({ "server/completion",                  "-nocb, --no-cont-batching",                     "Disable continuous batching" });
    opts.push_back({ "server/completion",                  "       --mmproj FILE",                          "Path to a multimodal projector file for LLaVA" });
    if (llama_supports_mlock()) {
        opts.push_back({ "server/completion",              "       --mlock",                                "Force system to keep model in RAM rather than swapping or compressing" });
    }
    if (llama_supports_mmap()) {
        opts.push_back({ "server/completion",              "       --no-mmap",                              "Disable memory-map model, slower load but may reduce pageouts if not using mlock" });
        opts.push_back({ "server/completion",              "       --mmap",                                 "Enable memory-map model, faster load but may increase pageouts if not using mlock" });
    }
    opts.push_back({ "server/completion",                  "       --numa TYPE",                            "Attempt optimizations that help on some NUMA systems\n"
                                                                                                            "  - distribute: spread execution evenly over all nodes\n"
                                                                                                            "  - isolate: only spawn threads on CPUs on the node that execution started on\n"
                                                                                                            "  - numactl: use the CPU map provided by numactl\n"
                                                                                                            "If run without this previously, it is recommended to drop the system page cache before using this, see https://github.com/ggerganov/llama.cpp/issues/1437" });
    opts.push_back({ "server/completion",                  "       --control-vector FILE",                  "Add a control vector" });
    opts.push_back({ "server/completion",                  "       --control-vector-scaled FILE SCALE",     "Add a control vector with user defined scaling SCALE" });
    opts.push_back({ "server/completion",                  "       --control-vector-layer-range START END", "Layer range to apply the control vector(s) to, start and end inclusive" });
    opts.push_back({ "server/completion",                  "       --spm-infill",                           "Use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this (default: %s)", llm_params.spm_infill ? "enabled" : "disabled" });
    opts.push_back({ "server/completion",                  "-sp,   --special",                              "Special tokens output enabled (default: %s)", llm_params.special ? "true" : "false" });
    // server // completion //
    // server // completion // speculative //
    opts.push_back({ "server/completion/speculative" });
    opts.push_back({ "server/completion/speculative",      "       --draft-max, --draft, --draft-n N",      "Number of tokens to draft for speculative decoding (default: %d)", llm_params.speculative.n_max });
    opts.push_back({ "server/completion/speculative",      "       --draft-min, --draft-n-min N",           "Minimum number of draft tokens to use for speculative decoding (default: %d)", llm_params.speculative.n_min });
    opts.push_back({ "server/completion/speculative",      "       --draft-p-min N",                        "Minimum speculative decoding probability (greedy) (default: %.1f)", llm_params.speculative.p_min });
    opts.push_back({ "server/completion/speculative",      "-md,   --model-draft FNAME",                    "Draft model for speculative decoding (default: unused)" });
    opts.push_back({ "server/completion/speculative",      "-devd, --device-draft <dev1,dev2,...>",         "A comma-separated list of devices to use for offloading the draft model (none = don't offload)\n"
                                                                                                            "Use --list-devices to see a list of available devices" });
    opts.push_back({ "server/completion/speculative",      "-ngld, --gpu-layers-draft, --n-gpu-layers-draft N",
                                                                                                            "Number of layers to store in VRAM for the draft model" });
    opts.push_back({ "server/completion/speculative",      "       --lookup-ngram-min N",                   "Minimum n-gram size for lookup cache (default: %d, 0 = disabled)", params_.lookup_ngram_min });
    opts.push_back({ "server/completion/speculative",      "-lcs,  --lookup-cache-static FILE",             "Path to static lookup cache to use for lookup decoding (not updated by generation)" });
    opts.push_back({ "server/completion/speculative",      "-lcd,  --lookup-cache-dynamic FILE",            "Path to dynamic lookup cache to use for lookup decoding (updated by generation)" });
    // server // completion // speculative //
    // server // completion // visual //
    opts.push_back({ "server/completion/visual" });
    opts.push_back({ "server/completion/visual",           "       --visual-max-image-size N",              "Maximum image size when completion with vision, resize the image size automatically if exceed, must be larger than 224 and be multiples of 14 (default: %d, 0 = disabled)", params_.max_image_size});
    // server // completion // visual //
    // server // embedding //
    opts.push_back({ "server/embedding" });
    opts.push_back({ "server/embedding",                   "       --pooling {none,mean,cls,last,rank}",    "Pooling type for embeddings, use model default if unspecified" });
    // server // embedding //
    // server // images //
    opts.push_back({ "server/images" });
    opts.push_back({ "server/images",                      "       --image-strength N",                     "Strength for noising, range of [0.0, 1.0], automatically retrieve the default value according to --model" });
    opts.push_back({ "server/images",                      "       --image-sampling-steps, --image-sample-steps N",
                                                                                                            "Number of sampling steps, automatically retrieve the default value according to --model, and +2 when requesting high definition generation" });
    opts.push_back({ "server/images",                      "       --image-cfg-scale N",                    "The scale of classifier-free guidance(CFG), automatically retrieve the default value according to --model (1.0 = disabled)" });
    opts.push_back({ "server/images",                      "       --image-slg-scale N",                    "The scale of skip-layer guidance(SLG), only for DiT model, automatically retrieve the default value according to --model (0.0 = disabled)" });
    opts.push_back({ "server/images",                      "       --image-slg-skip-layer",                 "The layers to skip when processing SLG, may be specified multiple times. (default: 7;8;9)" });

    opts.push_back({ "server/images",                      "       --image-no-text-encoder-model-offload",  "Disable text-encoder(clip-l/clip-g/t5xxl) model offload" });
    opts.push_back({ "server/images",                      "       --image-clip-l-model PATH",              "Path to the CLIP Large (clip-l) text encoder, or use --model included" });
    opts.push_back({ "server/images",                      "       --image-clip-g-model PATH",              "Path to the CLIP Generic (clip-g) text encoder, or use --model included" });
    opts.push_back({ "server/images",                      "       --image-t5xxl-model PATH",               "Path to the Text-to-Text Transfer Transformer (t5xxl) text encoder, or use --model included" });
    opts.push_back({ "server/images",                      "       --image-no-vae-model-offload",           "Disable vae(taesd) model offload" });
    opts.push_back({ "server/images",                      "       --image-vae-model PATH",                 "Path to Variational AutoEncoder (vae), or use --model included" });
    
    opts.push_back({ "server/images",                      "       --image-no-vae-tiling",                  "Disable vae decoder in tiles" });
    opts.push_back({ "server/images",                      "       --image-taesd-model PATH",               "Path to Tiny AutoEncoder For StableDiffusion (taesd), or use --model included" });
    opts.push_back({ "server/images",                      "       --image-upscale-model PATH",             "Path to the upscale model, or use --model included" });
    
    opts.push_back({ "server/images",                      "       --image-no-control-net-model-offload",   "Disable control-net model offload" });
    opts.push_back({ "server/images",                      "       --image-control-net-model PATH",         "Path to the control net model, or use --model included" });

    // clang-format on

    printf("usage: %s [options]\n", argv[0]);

    for (const auto &o : opts) {
        if (!o.grp.empty()) {
            printf("\n%s:\n\n", o.grp.c_str());
            continue;
        }
        printf("  %-32s", o.args.c_str());
        if (o.args.length() > 30) {
            printf("\n%34s", "");
        }

        const auto desc = o.desc;
        size_t start    = 0;
        size_t end      = desc.find('\n');
        while (end != std::string::npos) {
            printf("%s\n%34s", desc.substr(start, end - start).c_str(), "");
            start = end + 1;
            end   = desc.find('\n', start);
        }

        printf("%s\n", desc.substr(start).c_str());
    }
    printf("\n");
}



static bool llama_box_params_parse(int argc, char **argv, llama_box_params &params_) {
    // load dynamic backends
    ggml_backend_load_all();

    try {
        for (int i = 1; i < argc;) {
            const char *flag = argv[i++];

            if (*flag != '-') {
                continue;
            }

            // general //

            if (!strcmp(flag, "-h") || !strcmp(flag, "--help") || !strcmp(flag, "--usage")) {
                llama_box_params_print_usage(argc, argv, params_);
                exit(0);
            }

            if (!strcmp(flag, "--version")) {
                fprintf(stderr, "version    : %d (%s)\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
                fprintf(stderr, "compiler   : %s\n", LLAMA_COMPILER);
                fprintf(stderr, "target     : %s\n", LLAMA_BUILD_TARGET);
                exit(0);
            }

            if (!strcmp(flag, "--system-info")) {
                fprintf(stderr, "system_info: %s\n", llama_print_system_info());
                exit(0);
            }

            if (!strcmp(flag, "--list-devices")) {
                std::vector<ggml_backend_dev_t> rpc_devices;
                std::vector<ggml_backend_dev_t> all_devices;
                for (size_t j = 0; j < ggml_backend_dev_count(); ++j) {
                    ggml_backend_device *dev = ggml_backend_dev_get(j);
                    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                        if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                            rpc_devices.push_back(dev);
                        } else {
                            all_devices.push_back(dev);
                        }
                    }
                }
                // insert RPC devices in front
                all_devices.insert(all_devices.begin(), rpc_devices.begin(), rpc_devices.end());
                fprintf(stderr, "available devices:\n");
                for (size_t j = 0; j < all_devices.size(); ++j) {
                    ggml_backend_device *dev = all_devices[j];
                    size_t free, total;
                    ggml_backend_dev_memory(dev, &free, &total);
                    fprintf(stderr, "  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total >> 20, free >> 20);
                }
                exit(0);
            }

            if (!strcmp(flag, "-v") || !strcmp(flag, "--verbose") || !strcmp(flag, "--log-verbose")) {
                params_.llm_params.verbosity = INT_MAX;
                common_log_set_verbosity_thold(INT_MAX);
                continue;
            }

            if (!strcmp(flag, "-lv") || !strcmp(flag, "--verbosity") || !strcmp(flag, "--log-verbosity")) {
                if (i == argc) {
                    missing("--log-verbosity");
                }
                char *arg                    = argv[i++];
                params_.llm_params.verbosity = std::stoi(std::string(arg));
                common_log_set_verbosity_thold(params_.llm_params.verbosity);
                continue;
            }

            if (!strcmp(flag, "--log-colors")) {
                common_log_set_colors(common_log_main(), true);
                continue;
            }

            // general //

            // server //

            if (!strcmp(flag, "--host")) {
                if (i == argc) {
                    missing("--host");
                }
                char *arg                   = argv[i++];
                params_.llm_params.hostname = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "--port")) {
                if (i == argc) {
                    missing("--port");
                }
                char *arg               = argv[i++];
                params_.llm_params.port = std::stoi(std::string(arg));
                if (params_.llm_params.port < 0 || params_.llm_params.port > 65535) {
                    invalid("--port");
                }
                continue;
            }

            if (!strcmp(flag, "-to") || !strcmp(flag, "--timeout")) {
                if (i == argc) {
                    missing("--timeout");
                }
                char *arg                        = argv[i++];
                params_.llm_params.timeout_read  = std::stoi(std::string(arg));
                params_.llm_params.timeout_write = params_.llm_params.timeout_read;
                continue;
            }

            if (!strcmp(flag, "--threads-http")) {
                if (i == argc) {
                    missing("--threads-http");
                }
                char *arg                         = argv[i++];
                params_.llm_params.n_threads_http = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--conn-idle")) { // extend
                if (i == argc) {
                    missing("--conn-idle");
                }
                char *arg         = argv[i++];
                params_.conn_idle = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--conn-keepalive")) { // extend
                if (i == argc) {
                    missing("--conn-keepalive");
                }
                char *arg              = argv[i++];
                params_.conn_keepalive = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-m") || !strcmp(flag, "--model")) {
                if (i == argc) {
                    missing("--model");
                }
                char *arg                = argv[i++];
                params_.llm_params.model = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "-a") || !strcmp(flag, "--alias")) {
                if (i == argc) {
                    missing("--alias");
                }
                char *arg                      = argv[i++];
                params_.llm_params.model_alias = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "--lora")) {
                if (i == argc) {
                    missing("--lora");
                }
                char *arg = argv[i++];
                params_.llm_params.lora_adapters.push_back({
                    std::string(arg),
                    1.0f,
                });
                continue;
            }

            if (!strcmp(flag, "--lora-scaled")) {
                if (i == argc) {
                    missing("--lora-scaled");
                }
                char *n = argv[i++];
                if (i == argc) {
                    invalid("--lora-scaled");
                }
                char *s = argv[i++];
                params_.llm_params.lora_adapters.push_back({
                    std::string(n),
                    std::stof(std::string(s)),
                });
                continue;
            }

            if (!strcmp(flag, "--lora-init-without-apply")) {
                params_.llm_params.lora_init_without_apply = true;
                continue;
            }

            if (!strcmp(flag, "-s") || !strcmp(flag, "--seed")) {
                if (i == argc) {
                    missing("--seed");
                }
                char *arg                        = argv[i++];
                params_.llm_params.sampling.seed = std::stoul(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-fa") || !strcmp(flag, "--flash-attn")) {
                params_.llm_params.flash_attn = true;
                continue;
            }

            if (!strcmp(flag, "--no-flash-attn")) {
                params_.llm_params.flash_attn = false;
                continue;
            }

            if (!strcmp(flag, "--metrics")) {
                params_.llm_params.endpoint_metrics = true;
                continue;
            }

            if (!strcmp(flag, "--infill")) {
                params_.endpoint_infill = true;
                continue;
            }

            if (!strcmp(flag, "--embedding") || !strcmp(flag, "--embeddings")) {
                params_.llm_params.embedding = true;
                continue;
            }

            if (!strcmp(flag, "--images")) {
                params_.endpoint_images = true;
                continue;
            }

            if (!strcmp(flag, "--reranking") || !strcmp(flag, "--rerank")) {
                params_.llm_params.reranking = true;
                continue;
            }

            if (!strcmp(flag, "--slots")) {
                params_.llm_params.endpoint_slots = true;
                continue;
            }

           

            if (!strcmp(flag, "-ts") || !strcmp(flag, "--tensor-split")) {
                if (i == argc) {
                    missing("--tensor-split");
                }
                char *arg = argv[i++];
                const std::regex regex{R"([,/]+)"};
                std::string arg_s{arg};
                std::sregex_token_iterator it{arg_s.begin(), arg_s.end(), regex, -1};
                std::vector<std::string> split_arg{it, {}};
                if (split_arg.size() >= llama_max_devices()) {
                    invalid("--tensor-split exceeds the number of devices");
                }
                for (size_t j = 0; j < llama_max_devices(); ++j) {
                    if (j < split_arg.size()) {
                        params_.llm_params.tensor_split[j] = std::stof(split_arg[j]);
                    } else {
                        params_.llm_params.tensor_split[j] = 0.0f;
                    }
                }
                continue;
            }

            if (!strcmp(flag, "-ngl") || !strcmp(flag, "--gpu-layers") || !strcmp(flag, "--n-gpu-layers")) {
                if (i == argc) {
                    missing("--gpu-layers");
                }
                char *arg                       = argv[i++];
                params_.llm_params.n_gpu_layers = std::stoi(arg);
                continue;
            }

            if (!strcmp(flag, "--no-warmup")) {
                params_.llm_params.warmup = false;
                continue;
            }

            if (!strcmp(flag, "--warmup")) {
                params_.llm_params.warmup = true;
                continue;
            }

            // server // completion//

            if (!strcmp(flag, "-devd") || !strcmp(flag, "--device-draft")) {
                if (i == argc) {
                    missing("--device-draft");
                }
                char *arg                              = argv[i++];
                params_.llm_params.speculative.devices = parse_device_list(arg);
                continue;
            }

            if (!strcmp(flag, "-sm") || !strcmp(flag, "--split-mode")) {
                if (i == argc) {
                    missing("--split-mode");
                }
                char *arg = argv[i++];
                if (!strcmp(arg, "none")) {
                    params_.llm_params.split_mode = LLAMA_SPLIT_MODE_NONE;
                } else if (!strcmp(arg, "layer")) {
                    params_.llm_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
                } else if (!strcmp(arg, "row")) {
                    params_.llm_params.split_mode = LLAMA_SPLIT_MODE_ROW;
                } else {
                    invalid("--split-mode");
                }
                continue;
            }

            if (!strcmp(flag, "-mg") || !strcmp(flag, "--main-gpu")) {
                if (i == argc) {
                    missing("--main-gpu");
                }
                char *arg                   = argv[i++];
                params_.llm_params.main_gpu = std::stoi(std::string(arg));
                if (params_.llm_params.main_gpu < 0 || params_.llm_params.main_gpu >= int32_t(llama_max_devices())) {
                    invalid("--main-gpu");
                }
                continue;
            }

            if (!strcmp(flag, "--override-kv")) {
                if (i == argc) {
                    missing("--override-kv");
                }
                char *arg = argv[i++];
                if (!string_parse_kv_override(arg, params_.llm_params.kv_overrides)) {
                    invalid("--override-kv");
                }
                continue;
            }

            if (!strcmp(flag, "--slot-save-path")) {
                if (i == argc) {
                    missing("--slot-save-path");
                }
                char *arg = argv[i++];
                if (arg[0] == '\0') {
                    invalid("--slot-save-path");
                }
                std::string p(arg);
                if (p[p.size() - 1] != DIRECTORY_SEPARATOR) {
                    p += DIRECTORY_SEPARATOR;
                }
                params_.llm_params.slot_save_path = p;
                continue;
            }

            if (!strcmp(flag, "--chat-template")) {
                if (i == argc) {
                    missing("--chat-template");
                }
                char *arg = argv[i++];
                if (arg[0] == '\0') {
                    invalid("--chat-template");
                }
                std::string t(arg);
                std::vector<const char *> tmpls = get_builtin_chat_templates();
                if (std::find(tmpls.begin(), tmpls.end(), t) == tmpls.end()) {
                    invalid("--chat-template, use one of the built-in templates");
                }
                params_.llm_params.chat_template = t;
                params_.llm_params.use_jinja     = false;
                continue;
            }

            if (!strcmp(flag, "--jinja")) {
                params_.llm_params.chat_template = "";
                params_.llm_params.use_jinja     = true;
                continue;
            }

            if (!strcmp(flag, "--chat-template-file")) {
                if (i == argc) {
                    missing("--chat-template-file");
                }
                if (!params_.llm_params.use_jinja) {
                    invalid("--chat-template-file, --jinja must be set before");
                }
                char *arg = argv[i++];
                std::ifstream file(arg);
                if (!file) {
                    invalid("--chat-template-file, failed to open file");
                }
                std::string t;
                std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), std::back_inserter(t));
                std::vector<const char *> tmpls = get_builtin_chat_templates();
                if (std::find(tmpls.begin(), tmpls.end(), t) != tmpls.end()) {
                    invalid("--chat-template-file, set --chat-template directly if using a built-in template");
                }
                params_.llm_params.chat_template = t;
                continue;
            }

            if (!strcmp(flag, "-sps") || !strcmp(flag, "--slot-prompt-similarity")) {
                if (i == argc) {
                    missing("--slot-prompt-similarity");
                }
                char *arg                                 = argv[i++];
                params_.llm_params.slot_prompt_similarity = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-tps") || !strcmp(flag, "--tokens-per-second")) { // extend
                if (i == argc) {
                    missing("--tokens-per-second");
                }
                char *arg     = argv[i++];
                params_.n_tps = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-t") || !strcmp(flag, "--threads")) {
                if (i == argc) {
                    missing("--threads");
                }
                char *arg                              = argv[i++];
                params_.llm_params.cpuparams.n_threads = std::stoi(std::string(arg));
                if (params_.llm_params.cpuparams.n_threads <= 0) {
                    params_.llm_params.cpuparams.n_threads = cpu_get_num_math();
                }
                continue;
            }

            if (!strcmp(flag, "-C") || !strcmp(flag, "--cpu-mask")) {
                if (i == argc) {
                    missing("--cpu-mask");
                }
                char *arg                               = argv[i++];
                params_.llm_params.cpuparams.mask_valid = true;
                if (!parse_cpu_mask(arg, params_.llm_params.cpuparams.cpumask)) {
                    invalid("--cpu-mask");
                }
                continue;
            }

            if (!strcmp(flag, "-Cr") || !strcmp(flag, "--cpu-range")) {
                if (i == argc) {
                    missing("--cpu-range");
                }
                char *arg                               = argv[i++];
                params_.llm_params.cpuparams.mask_valid = true;
                if (!parse_cpu_range(arg, params_.llm_params.cpuparams.cpumask)) {
                    invalid("--cpu-range");
                }
                continue;
            }

            if (!strcmp(flag, "--prio")) {
                if (i == argc) {
                    missing("--prio");
                }
                char *arg                             = argv[i++];
                params_.llm_params.cpuparams.priority = (enum ggml_sched_priority)std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--cpu-strict")) {
                if (i == argc) {
                    missing("--cpu-strict");
                }
                char *arg                               = argv[i++];
                params_.llm_params.cpuparams.strict_cpu = std::stoul(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--poll")) {
                if (i == argc) {
                    missing("--poll");
                }
                char *arg                         = argv[i++];
                params_.llm_params.cpuparams.poll = std::stoul(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-tb") || !strcmp(flag, "--threads-batch")) {
                if (i == argc) {
                    missing("--threads-batch");
                }
                char *arg                                    = argv[i++];
                params_.llm_params.cpuparams_batch.n_threads = std::stoi(std::string(arg));
                if (params_.llm_params.cpuparams_batch.n_threads <= 0) {
                    params_.llm_params.cpuparams_batch.n_threads = cpu_get_num_math();
                }
                continue;
            }

            if (!strcmp(flag, "-Cb") || !strcmp(flag, "--cpu-mask-batch")) {
                if (i == argc) {
                    missing("--cpu-mask-batch");
                }
                char *arg                                     = argv[i++];
                params_.llm_params.cpuparams_batch.mask_valid = true;
                if (!parse_cpu_mask(arg, params_.llm_params.cpuparams_batch.cpumask)) {
                    invalid("--cpu-mask-batch");
                }
                continue;
            }

            if (!strcmp(flag, "-Crb") || !strcmp(flag, "--cpu-range-batch")) {
                if (i == argc) {
                    missing("--cpu-range-batch");
                }
                char *arg                                     = argv[i++];
                params_.llm_params.cpuparams_batch.mask_valid = true;
                if (!parse_cpu_range(arg, params_.llm_params.cpuparams_batch.cpumask)) {
                    invalid("--cpu-range-batch");
                }
                continue;
            }

            if (!strcmp(flag, "--prio-batch")) {
                if (i == argc) {
                    missing("--prio-batch");
                }
                char *arg                                   = argv[i++];
                params_.llm_params.cpuparams_batch.priority = (enum ggml_sched_priority)std::stoul(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--cpu-strict-batch")) {
                if (i == argc) {
                    missing("--cpu-strict-batch");
                }
                char *arg                                     = argv[i++];
                params_.llm_params.cpuparams_batch.strict_cpu = std::stoul(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--poll-batch")) {
                if (i == argc) {
                    missing("--poll-batch");
                }
                char *arg                               = argv[i++];
                params_.llm_params.cpuparams_batch.poll = std::stoul(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-c") || !strcmp(flag, "--ctx-size")) {
                if (i == argc) {
                    missing("--ctx-size");
                }
                char *arg                = argv[i++];
                params_.llm_params.n_ctx = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--no-context-shift")) {
                params_.force_context_shift  = false;
                params_.llm_params.ctx_shift = false;
                continue;
            }

            if (!strcmp(flag, "--context-shift")) {
                params_.force_context_shift  = true;
                params_.llm_params.ctx_shift = true;
                continue;
            }

            if (!strcmp(flag, "-n") || !strcmp(flag, "--predict")) {
                if (i == argc) {
                    missing("--predict");
                }
                char *arg                    = argv[i++];
                params_.llm_params.n_predict = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-b") || !strcmp(flag, "--batch-size")) {
                if (i == argc) {
                    missing("--batch-size");
                }
                char *arg                  = argv[i++];
                params_.llm_params.n_batch = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-ub") || !strcmp(flag, "--ubatch-size")) {
                if (i == argc) {
                    missing("--ubatch-size");
                }
                char *arg                   = argv[i++];
                params_.llm_params.n_ubatch = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--keep")) {
                if (i == argc) {
                    missing("--keep");
                }
                char *arg                 = argv[i++];
                params_.llm_params.n_keep = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-e") || !strcmp(flag, "--escape")) {
                params_.llm_params.escape = true;
                continue;
            }

            if (!strcmp(flag, "--no-escape")) {
                params_.llm_params.escape = false;
                continue;
            }

            if (!strcmp(flag, "--samplers")) {
                if (i == argc) {
                    missing("--samplers");
                }
                char *arg                            = argv[i++];
                const auto sampler_names             = string_split<std::string>(arg, ';');
                params_.llm_params.sampling.samplers = common_sampler_types_from_names(sampler_names, true);
                continue;
            }

            if (!strcmp(flag, "--sampling-seq")) {
                if (i == argc) {
                    missing("--sampling-seq");
                }
                char *arg                            = argv[i++];
                params_.llm_params.sampling.samplers = common_sampler_types_from_chars(arg);
                continue;
            }

            if (!strcmp(flag, "--temp")) {
                if (i == argc) {
                    missing("--temp");
                }
                char *arg                        = argv[i++];
                params_.llm_params.sampling.temp = std::stof(std::string(arg));
                params_.llm_params.sampling.temp = std::max(params_.llm_params.sampling.temp, 0.0f);
                continue;
            }

            if (!strcmp(flag, "--top-k")) {
                if (i == argc) {
                    missing("--top-k");
                }
                char *arg                         = argv[i++];
                params_.llm_params.sampling.top_k = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--top-p")) {
                if (i == argc) {
                    missing("--top-p");
                }
                char *arg                         = argv[i++];
                params_.llm_params.sampling.top_p = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--min-p")) {
                if (i == argc) {
                    missing("--min-p");
                }
                char *arg                         = argv[i++];
                params_.llm_params.sampling.min_p = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--top-nsigma")) {
                if (i == argc) {
                    missing("--top-nsigma");
                }
                char *arg                               = argv[i++];
                params_.llm_params.sampling.top_n_sigma = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--xtc-probability")) {
                if (i == argc) {
                    missing("--xtc-probability");
                }
                char *arg                                   = argv[i++];
                params_.llm_params.sampling.xtc_probability = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--xtc-threshold")) {
                if (i == argc) {
                    missing("--xtc-threshold");
                }
                char *arg                                 = argv[i++];
                params_.llm_params.sampling.xtc_threshold = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--typical")) {
                if (i == argc) {
                    missing("--typical");
                }
                char *arg                         = argv[i++];
                params_.llm_params.sampling.typ_p = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--repeat-last-n")) {
                if (i == argc) {
                    missing("--repeat-last-n");
                }
                char *arg                                  = argv[i++];
                params_.llm_params.sampling.penalty_last_n = std::stoi(std::string(arg));
                if (params_.llm_params.sampling.penalty_last_n < -1) {
                    invalid("--repeat-last-n");
                }
                params_.llm_params.sampling.n_prev = std::max(params_.llm_params.sampling.n_prev, params_.llm_params.sampling.penalty_last_n);
                continue;
            }

            if (!strcmp(flag, "--repeat-penalty")) {
                if (i == argc) {
                    missing("--repeat-penalty");
                }
                char *arg                                  = argv[i++];
                params_.llm_params.sampling.penalty_repeat = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--presence-penalty")) {
                if (i == argc) {
                    missing("--presence-penalty");
                }
                char *arg                                   = argv[i++];
                params_.llm_params.sampling.penalty_present = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--frequency-penalty")) {
                if (i == argc) {
                    missing("--frequency-penalty");
                }
                char *arg                                = argv[i++];
                params_.llm_params.sampling.penalty_freq = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--dry-multiplier")) {
                if (i == argc) {
                    missing("--dry-multiplier");
                }
                char *arg                                  = argv[i++];
                params_.llm_params.sampling.dry_multiplier = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--dry-base")) {
                if (i == argc) {
                    missing("--dry-base");
                }
                char *arg            = argv[i++];
                float potential_base = std::stof(std::string(arg));
                if (potential_base >= 1.0f) {
                    params_.llm_params.sampling.dry_multiplier = std::stof(std::string(arg));
                }
                continue;
            }

            if (!strcmp(flag, "--dry-allowed-length")) {
                if (i == argc) {
                    missing("--dry-allowed-length");
                }
                char *arg                                      = argv[i++];
                params_.llm_params.sampling.dry_allowed_length = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--dry-penalty-last-n")) {
                if (i == argc) {
                    missing("--dry-penalty-last-n");
                }
                char *arg                                      = argv[i++];
                params_.llm_params.sampling.dry_penalty_last_n = std::stoi(std::string(arg));
                if (params_.llm_params.sampling.dry_penalty_last_n < -1) {
                    invalid("--dry-penalty-last-n");
                }
                continue;
            }

            if (!strcmp(flag, "--dry-sequence-breaker")) {
                if (i == argc) {
                    missing("--dry-sequence-breaker");
                }

                static bool defaults_cleared = false;
                if (!defaults_cleared) {
                    params_.llm_params.sampling.dry_sequence_breakers.clear();
                    defaults_cleared = true;
                }

                char *arg = argv[i++];
                if (!strcmp(arg, "none")) {
                    params_.llm_params.sampling.dry_sequence_breakers.clear();
                } else {
                    params_.llm_params.sampling.dry_sequence_breakers.emplace_back(arg);
                }
                continue;
            }

            if (!strcmp(flag, "--dynatemp-range")) {
                if (i == argc) {
                    missing("--dynatemp-range");
                }
                char *arg                                  = argv[i++];
                params_.llm_params.sampling.dynatemp_range = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--dynatemp-exp")) {
                if (i == argc) {
                    missing("--dynatemp-exp");
                }
                char *arg                                     = argv[i++];
                params_.llm_params.sampling.dynatemp_exponent = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--mirostat")) {
                if (i == argc) {
                    missing("--mirostat");
                }
                char *arg                            = argv[i++];
                params_.llm_params.sampling.mirostat = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--mirostat-lr")) {
                if (i == argc) {
                    missing("--mirostat-lr");
                }
                char *arg                                = argv[i++];
                params_.llm_params.sampling.mirostat_eta = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--mirostat-ent")) {
                if (i == argc) {
                    missing("--mirostat-ent");
                }
                char *arg                                = argv[i++];
                params_.llm_params.sampling.mirostat_tau = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-l") || !strcmp(flag, "--logit-bias")) {
                if (i == argc) {
                    missing("--logit-bias");
                }
                char *arg = argv[i++];
                std::stringstream ss(arg);
                llama_token key;
                char sign;
                std::string value;
                if (ss >> key && ss >> sign && std::getline(ss, value) && (sign == '+' || sign == '-')) {
                    const float bias = std::stof(value) * ((sign == '-') ? -1.0f : 1.0f);
                    params_.llm_params.sampling.logit_bias.push_back({key, bias});
                } else {
                    invalid("--logit-bias");
                }
                continue;
            }

            if (!strcmp(flag, "--grammar")) {
                if (i == argc) {
                    missing("--grammar");
                }
                char *arg                           = argv[i++];
                params_.llm_params.sampling.grammar = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "--grammar-file")) {
                if (i == argc) {
                    missing("--grammar-file");
                }
                char *arg = argv[i++];
                std::ifstream file(arg);
                if (!file) {
                    invalid("--grammar-file, failed to open file");
                }
                std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(),
                          std::back_inserter(params_.llm_params.sampling.grammar));
                continue;
            }

            if (!strcmp(flag, "-j") || !strcmp(flag, "--json-schema")) {
                if (i == argc) {
                    missing("--json-schema");
                }
                char *arg                           = argv[i++];
                params_.llm_params.sampling.grammar = json_schema_to_grammar(json::parse(std::string(arg)));
                continue;
            }

            if (!strcmp(flag, "--rope-scaling")) {
                if (i == argc) {
                    missing("--rope-scaling");
                }
                char *arg = argv[i++];
                std::string value(arg);
                if (value == "none") {
                    params_.llm_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
                } else if (value == "linear") {
                    params_.llm_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
                } else if (value == "yarn") {
                    params_.llm_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
                } else {
                    invalid("--rope-scaling");
                }
                continue;
            }

            if (!strcmp(flag, "--rope-scale")) {
                if (i == argc) {
                    missing("--rope-scale");
                }
                char *arg                          = argv[i++];
                params_.llm_params.rope_freq_scale = 1.0f / std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--rope-freq-base")) {
                if (i == argc) {
                    missing("--rope-freq-base");
                }
                char *arg                         = argv[i++];
                params_.llm_params.rope_freq_base = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--rope-freq-scale")) {
                if (i == argc) {
                    missing("--rope-freq-scale");
                }
                char *arg                          = argv[i++];
                params_.llm_params.rope_freq_scale = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--yarn-orig-ctx")) {
                if (i == argc) {
                    missing("--yarn-orig-ctx");
                }
                char *arg                        = argv[i++];
                params_.llm_params.yarn_orig_ctx = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--yarn-ext-factor")) {
                if (i == argc) {
                    missing("--yarn-ext-factor");
                }
                char *arg                          = argv[i++];
                params_.llm_params.yarn_ext_factor = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--yarn-attn-factor")) {
                if (i == argc) {
                    missing("--yarn-attn-factor");
                }
                char *arg                           = argv[i++];
                params_.llm_params.yarn_attn_factor = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--yarn-beta-fast")) {
                if (i == argc) {
                    missing("--yarn-beta-fast");
                }
                char *arg                         = argv[i++];
                params_.llm_params.yarn_beta_fast = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--yarn-beta-slow")) {
                if (i == argc) {
                    missing("--yarn-beta-slow");
                }
                char *arg                         = argv[i++];
                params_.llm_params.yarn_beta_slow = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-nkvo") || !strcmp(flag, "--no-kv-offload")) {
                params_.llm_params.no_kv_offload = true;
                continue;
            }

            if (!strcmp(flag, "--no-cache-prompt")) {
                params_.cache_prompt = false;
                continue;
            }

            if (!strcmp(flag, "--cache-reuse")) {
                if (i == argc) {
                    missing("--cache-reuse");
                }
                char *arg                        = argv[i++];
                params_.llm_params.n_cache_reuse = std::stoi(std::string(arg));
                if (params_.llm_params.n_cache_reuse > 0) {
                    params_.cache_prompt = true;
                }
                continue;
            }

            if (!strcmp(flag, "-ctk") || !strcmp(flag, "--cache-type-k")) {
                if (i == argc) {
                    missing("--cache-type-k");
                }
                char *arg                       = argv[i++];
                params_.llm_params.cache_type_k = parse_cache_kv_type(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-ctv") || !strcmp(flag, "--cache-type-v")) {
                if (i == argc) {
                    missing("--cache-type-v");
                }
                char *arg                       = argv[i++];
                params_.llm_params.cache_type_v = parse_cache_kv_type(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-dt") || !strcmp(flag, "--defrag-thold")) {
                if (i == argc) {
                    missing("--defrag-thold");
                }
                char *arg                       = argv[i++];
                params_.llm_params.defrag_thold = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-np") || !strcmp(flag, "--parallel")) {
                if (i == argc) {
                    missing("--parallel");
                }
                char *arg                     = argv[i++];
                params_.llm_params.n_parallel = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-nocb") || !strcmp(flag, "--no-cont-batching")) {
                params_.llm_params.cont_batching = false;
                continue;
            }

            if (!strcmp(flag, "--mmproj")) {
                if (i == argc) {
                    missing("--mmproj");
                }
                char *arg                 = argv[i++];
                params_.llm_params.mmproj = std::string(arg);
                continue;
            }

            if (llama_supports_mlock()) {
                if (!strcmp(flag, "--mlock")) {
                    params_.llm_params.use_mlock = true;
                    continue;
                }
            }

            if (llama_supports_mmap()) {
                if (!strcmp(flag, "--no-mmap")) {
                    params_.llm_params.use_mmap = false;
                    continue;
                }
                if (!strcmp(flag, "--mmap")) {
                    params_.llm_params.use_mmap = true;
                    continue;
                }
            }

            if (!strcmp(flag, "--numa")) {
                if (i == argc) {
                    missing("--numa");
                }
                char *arg = argv[i++];
                std::string value(arg);
                if (value == "distribute") {
                    params_.llm_params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE;
                } else if (value == "isolate") {
                    params_.llm_params.numa = GGML_NUMA_STRATEGY_ISOLATE;
                } else if (value == "numactl") {
                    params_.llm_params.numa = GGML_NUMA_STRATEGY_NUMACTL;
                } else {
                    invalid("--numa");
                }
                continue;
            }

            if (!strcmp(flag, "--control-vector")) {
                if (i == argc) {
                    missing("--control-vector");
                }
                char *arg = argv[i++];
                params_.llm_params.control_vectors.push_back({1.0f, std::string(arg)});
                continue;
            }

            if (!strcmp(flag, "--control-vector-scaled")) {
                if (i == argc) {
                    missing("--control-vector-scaled");
                }
                char *n = argv[i++];
                if (i == argc) {
                    invalid("--control-vector-scaled");
                }
                char *s = argv[i++];
                params_.llm_params.control_vectors.push_back({std::stof(std::string(s)), std::string(n)});
                continue;
            }

            if (!strcmp(flag, "--control-vector-layer-range")) {
                if (i == argc) {
                    missing("--control-vector-layer-range");
                }
                char *s = argv[i++];
                if (i == argc) {
                    invalid("--control-vector-layer-range");
                }
                char *e                                       = argv[i++];
                params_.llm_params.control_vector_layer_start = std::stoi(std::string(s));
                params_.llm_params.control_vector_layer_end   = std::stoi(std::string(e));
                continue;
            }

            if (!strcmp(flag, "--spm-infill")) {
                params_.llm_params.spm_infill = true;
                continue;
            }

            if (!strcmp(flag, "-sp") || !strcmp(flag, "--special")) {
                params_.llm_params.special = true;
                continue;
            }

            // server // completion // speculative //

            if (!strcmp(flag, "--draft") || !strcmp(flag, "--draft-max") || !strcmp(flag, "--draft-n")) {
                if (i == argc) {
                    missing("--draft-max");
                }
                char *arg                            = argv[i++];
                params_.llm_params.speculative.n_max = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--draft-min") || !strcmp(flag, "--draft-n-min")) {
                if (i == argc) {
                    missing("--draft-min");
                }
                char *arg                            = argv[i++];
                params_.llm_params.speculative.n_min = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--draft-p-min")) {
                if (i == argc) {
                    missing("--draft-p-min");
                }
                char *arg                            = argv[i++];
                params_.llm_params.speculative.p_min = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-md") || !strcmp(flag, "--model-draft")) {
                if (i == argc) {
                    missing("--model-draft");
                }
                char *arg                            = argv[i++];
                params_.llm_params.speculative.model = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "-dev") || !strcmp(flag, "--device")) {
                if (i == argc) {
                    missing("--device");
                }
                char *arg                  = argv[i++];
                params_.llm_params.devices = parse_device_list(arg);
                continue;
            }

            if (!strcmp(flag, "-ngld") || !strcmp(flag, "--gpu-layers-draft") || !strcmp(flag, "--n-gpu-layers-draft")) {
                if (i == argc) {
                    missing("--gpu-layers-draft");
                }
                char *arg                                   = argv[i++];
                params_.llm_params.speculative.n_gpu_layers = std::stoi(arg);
                continue;
            }

            if (!strcmp(flag, "--lookup-ngram-min")) {
                if (i == argc) {
                    missing("--lookup-ngram-min");
                }
                char *arg                = argv[i++];
                params_.lookup_ngram_min = std::stoi(std::string(arg));
                if (params_.lookup_ngram_min < 1) {
                    invalid("--lookup-ngram-min");
                }
                if (params_.lookup_ngram_min > LLAMA_NGRAM_MAX) {
                    invalid("--lookup-ngram-min");
                }
                continue;
            }

            if (!strcmp(flag, "-lcs") || !strcmp(flag, "--lookup-cache-static")) {
                if (i == argc) {
                    missing("--lookup-cache-static");
                }
                char *arg                              = argv[i++];
                params_.llm_params.lookup_cache_static = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "-lcd") || !strcmp(flag, "--lookup-cache-dynamic")) {
                if (i == argc) {
                    missing("--lookup-cache-dynamic");
                }
                char *arg                               = argv[i++];
                params_.llm_params.lookup_cache_dynamic = std::string(arg);
                continue;
            }

            // server // completion // visual //

            if (!strcmp(flag, "--visual-max-image-size")) {
                if (i == argc) {
                    missing("--visual-max-image-size");
                }
                char *arg              = argv[i++];
                params_.max_image_size = std::stoi(std::string(arg));
                if (params_.max_image_size != 0 && params_.max_image_size < 224) {
                    invalid("--visual-max-image-size, must be at least 224");
                }
                if (params_.max_image_size % 14 != 0) {
                    invalid("--visual-max-image-size, must be a multiple of 14");
                }
                continue;
            }

            // server // embedding //

            if (!strcmp(flag, "--pooling")) {
                if (i == argc) {
                    missing("--pooling");
                }
                char *arg = argv[i++];
                std::string value(arg);
                if (value == "none") {
                    params_.llm_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
                } else if (value == "mean") {
                    params_.llm_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
                } else if (value == "cls") {
                    params_.llm_params.pooling_type = LLAMA_POOLING_TYPE_CLS;
                } else if (value == "last") {
                    params_.llm_params.pooling_type = LLAMA_POOLING_TYPE_LAST;
                } else if (value == "rank") {
                    params_.llm_params.pooling_type = LLAMA_POOLING_TYPE_RANK;
                } else {
                    invalid("--pooling");
                }
                continue;
            }

            unknown(flag);
        }
    } catch (const std::invalid_argument &ex) {
        fprintf(stderr, "%s\n", ex.what());
        return false;
    }

    // Retrieve params from environment variables
    get_env("LLAMA_ARG_MODEL", params_.llm_params.model);
    get_env("LLAMA_ARG_MODEL_ALIAS", params_.llm_params.model_alias);
    get_env("LLAMA_ARG_THREADS", params_.llm_params.cpuparams.n_threads);
    get_env("LLAMA_ARG_CTX_SIZE", params_.llm_params.n_ctx);
    get_env("LLAMA_ARG_N_PARALLEL", params_.llm_params.n_parallel);
    get_env("LLAMA_ARG_BATCH", params_.llm_params.n_batch);
    get_env("LLAMA_ARG_UBATCH", params_.llm_params.n_ubatch);
    get_env("LLAMA_ARG_DEVICE", params_.llm_params.devices);
    get_env("LLAMA_ARG_N_GPU_LAYERS", params_.llm_params.n_gpu_layers);
    get_env("LLAMA_ARG_THREADS_HTTP", params_.llm_params.n_threads_http);
    get_env("LLAMA_ARG_CACHE_PROMPT", params_.cache_prompt);
    get_env("LLAMA_ARG_CACHE_REUSE", params_.llm_params.n_cache_reuse);
    get_env("LLAMA_ARG_CHAT_TEMPLATE", params_.llm_params.chat_template);
    get_env("LLAMA_ARG_JINJA", params_.llm_params.use_jinja);
    get_env("LLAMA_ARG_N_PREDICT", params_.llm_params.n_predict);
    get_env("LLAMA_ARG_METRICS", params_.llm_params.endpoint_metrics);
    get_env("LLAMA_ARG_SLOTS", params_.llm_params.endpoint_slots);
    get_env("LLAMA_ARG_EMBEDDINGS", params_.llm_params.embedding);
    get_env("LLAMA_ARG_FLASH_ATTN", params_.llm_params.flash_attn);
    get_env("LLAMA_ARG_DEFRAG_THOLD", params_.llm_params.defrag_thold);
    get_env("LLAMA_ARG_CONT_BATCHING", params_.llm_params.cont_batching);
    get_env("LLAMA_ARG_HOST", params_.llm_params.hostname);
    get_env("LLAMA_ARG_PORT", params_.llm_params.port);
    get_env("LLAMA_ARG_DRAFT_MAX", params_.llm_params.speculative.n_max);
    get_env("LLAMA_ARG_DRAFT_MIN", params_.llm_params.speculative.n_min);
    get_env("LLAMA_ARG_DRAFT_P_MIN", params_.llm_params.speculative.p_min);
    get_env("LLAMA_ARG_MODEL_DRAFT", params_.llm_params.speculative.model);
    get_env("LLAMA_ARG_DEVICE_DRAFT", params_.llm_params.speculative.devices);
    get_env("LLAMA_ARG_N_GPU_LAYERS_DRAFT", params_.llm_params.speculative.n_gpu_layers);
    get_env("LLAMA_ARG_LOOKUP_NGRAM_MIN", params_.lookup_ngram_min);
    get_env("LLAMA_ARG_LOOKUP_CACHE_STATIC", params_.llm_params.lookup_cache_static);
    get_env("LLAMA_ARG_LOOKUP_CACHE_DYNAMIC", params_.llm_params.lookup_cache_dynamic);
    get_env("LLAMA_LOG_VERBOSITY", params_.llm_params.verbosity);

    // Postprocess params
    if (params_.llm_params.chat_template.size() > 20 && !common_chat_verify_template(params_.llm_params.chat_template, params_.llm_params.use_jinja)) {
        invalid("--chat-template");
    }
    postprocess_cpu_params(params_.llm_params.cpuparams, nullptr);
    postprocess_cpu_params(params_.llm_params.cpuparams_batch, &params_.llm_params.cpuparams);
    postprocess_cpu_params(params_.llm_params.speculative.cpuparams, &params_.llm_params.cpuparams);
    postprocess_cpu_params(params_.llm_params.speculative.cpuparams_batch, &params_.llm_params.cpuparams_batch);
    if (!params_.llm_params.devices.empty() && params_.llm_params.speculative.devices.empty()) {
        params_.llm_params.speculative.devices = params_.llm_params.devices;
    }

    if (!params_.llm_params.kv_overrides.empty()) {
        params_.llm_params.kv_overrides.emplace_back();
        params_.llm_params.kv_overrides.back().key[0] = 0;
    }

    if (params_.llm_params.lora_init_without_apply) {
        for (auto &lora_adapter : params_.llm_params.lora_adapters) {
            lora_adapter.scale = 0.0f;
        }
    }

    return true;
}