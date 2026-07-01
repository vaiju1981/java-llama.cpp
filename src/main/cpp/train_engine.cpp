// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#include "train_engine.h"

#include "common.h"
#include "ggml-opt.h"
#include "llama.h"

#include <jni.h>

#include <exception>
#include <string>
#include <vector>

namespace jllama_train {

bool finetune(const finetune_config &cfg, std::string &err) {
    common_params params;
    params.escape = false;
    params.model.path = cfg.model_path;
    params.prompt = cfg.training_text;
    params.out_file = cfg.output_path;
    params.n_ctx = cfg.n_ctx;
    params.n_gpu_layers = cfg.n_gpu_layers;
    params.lr.lr0 = cfg.learning_rate;
    params.lr.epochs = static_cast<unsigned>(cfg.epochs > 0 ? cfg.epochs : 1);
    params.lr.init(); // required after setting lr fields, before get_lr() is used by the optimizer

    // Training needs writable weights (mmap yields read-only pointers) and an f32 KV cache
    // (OUT_PROD has no f16 support) — same forced settings as upstream finetune.cpp.
    params.use_mmap = false;
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;

    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model *model = llama_init->model();
    llama_context *ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        err = "failed to load model for training: " + cfg.model_path;
        return false;
    }

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    if (tokens.size() < 2) {
        err = "training text produced too few tokens (need at least 2)";
        return false;
    }

    ggml_opt_dataset_t dataset = common_opt_dataset_init(ctx, tokens, llama_n_ctx(ctx) / 2);

    llama_opt_params lopt_params = {
        /*n_ctx_train     =*/0,
        /*param_filter    =*/llama_opt_param_filter_all,
        /*param_filter_ud =*/nullptr,
        /*get_opt_pars    =*/common_opt_lr_pars,
        /*get_opt_pars_ud =*/&params.lr,
        /*optimizer_type  =*/params.optimizer,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval = ggml_opt_result_init();

    for (params.lr.epoch = 0; params.lr.epoch < params.lr.epochs; ++params.lr.epoch) {
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                        ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);
    ggml_opt_dataset_free(dataset);

    llama_model_save_to_file(model, params.out_file.c_str());

    // Deliberately NOT calling llama_backend_free(): other live llama contexts in this JVM
    // (e.g. an inference LlamaModel) may still depend on the initialized backend.
    return true;
}

} // namespace jllama_train

extern "C" JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaTrainer_finetuneNative(
    JNIEnv *env, jclass, jstring jmodel, jstring jtext, jstring jout, jint epochs, jfloat learningRate,
    jint nCtx, jint nGpuLayers) {
    const auto to_str = [env](jstring s) -> std::string {
        if (s == nullptr) {
            return "";
        }
        const char *c = env->GetStringUTFChars(s, nullptr);
        std::string out = c != nullptr ? c : "";
        if (c != nullptr) {
            env->ReleaseStringUTFChars(s, c);
        }
        return out;
    };

    jllama_train::finetune_config cfg;
    cfg.model_path = to_str(jmodel);
    cfg.training_text = to_str(jtext);
    cfg.output_path = to_str(jout);
    cfg.epochs = static_cast<int>(epochs);
    cfg.learning_rate = static_cast<float>(learningRate);
    cfg.n_ctx = static_cast<int>(nCtx);
    cfg.n_gpu_layers = static_cast<int>(nGpuLayers);

    std::string err;
    try {
        if (jllama_train::finetune(cfg, err)) {
            return env->NewStringUTF(""); // empty == success
        }
    } catch (const std::exception &e) {
        err = e.what();
    } catch (...) {
        err = "unknown C++ exception during fine-tuning";
    }
    return env->NewStringUTF(err.c_str());
}
