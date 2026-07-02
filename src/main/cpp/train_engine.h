// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Native fine-tuning engine (proof-of-concept): a self-contained wrapper over llama.cpp's
// ggml-opt training path (llama_opt_init / llama_opt_epoch), mirroring upstream
// examples/training/finetune.cpp. Loads its own model + context (independent of the inference
// server_context in jllama.cpp), fine-tunes on a text corpus, and writes a new GGUF via
// llama_model_save_to_file. Kept out of jllama.cpp so the JNI layer stays thin.

#ifndef JLLAMA_TRAIN_ENGINE_H
#define JLLAMA_TRAIN_ENGINE_H

#include <string>

namespace jllama_train {

// One fine-tuning run's inputs.
struct finetune_config {
    std::string model_path;    // base GGUF to fine-tune
    std::string training_text; // corpus supplied inline (used when training_file is empty)
    std::string training_file; // corpus read from this path instead of training_text
    std::string output_path;   // where the fine-tuned GGUF is written
    int         epochs;        // number of passes over the corpus (>= 1)
    float       learning_rate; // lr at the first epoch
    float       lr_min;        // minimum lr for decay; < 0 = no decay
    float       decay_epochs;  // decay lr0 -> lr_min over this many epochs; <= 0 = disabled
    float       weight_decay;  // weight decay; 0 = disabled
    int         optimizer;     // ggml_opt_optimizer_type: 0 = AdamW, 1 = SGD
    int         n_ctx;         // context size; 0 = the model's trained context
    int         n_gpu_layers;  // layers offloaded to the GPU; -1 = auto
    float       val_split;     // fraction of the corpus held out for validation
    int         n_batch;       // logical batch size; 0 = native default
    int         n_ubatch;      // physical (micro) batch size; 0 = native default
};

// Run one fine-tuning job end to end. Returns true on success; on failure returns false and sets
// `err`. Not re-entrant; intended to be called off the JVM's critical threads (it blocks for the
// full training run).
bool finetune(const finetune_config &cfg, std::string &err);

} // namespace jllama_train

#endif // JLLAMA_TRAIN_ENGINE_H
