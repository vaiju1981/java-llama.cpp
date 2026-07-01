// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.nio.file.Path;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.loader.LlamaLoader;

/**
 * Proof-of-concept in-process fine-tuning entry point, wrapping llama.cpp's ggml-opt training path
 * ({@code llama_opt_init} / {@code llama_opt_epoch}) the same way the upstream
 * {@code examples/training/finetune.cpp} tool does. Loads its own model and context (independent of
 * {@link LlamaModel}), fine-tunes on a text corpus, and writes a new GGUF.
 *
 * <p><strong>Status: proof of concept.</strong> Full-model fine-tuning is compute- and
 * memory-intensive and blocks for the whole run; upstream training support is itself experimental.
 * This surface is intentionally minimal so the native path (which links ggml-opt into
 * {@code libjllama} with no extra dependency) can be exercised end to end before a richer
 * {@code FineTuner} API is designed.
 */
public final class LlamaTrainer {

    static {
        LlamaLoader.initialize();
    }

    private LlamaTrainer() {}

    /**
     * Fine-tune {@code model} on {@code trainingText} for {@code epochs} passes, writing the result
     * to {@code output}. Uses the model's trained context size and automatic GPU-layer selection.
     *
     * @param model the base GGUF model to fine-tune
     * @param trainingText the training corpus (tokenized in-process)
     * @param output the path the fine-tuned GGUF is written to
     * @param epochs number of passes over the corpus (at least 1)
     * @param learningRate the AdamW learning rate at the first epoch (e.g. {@code 1e-5f})
     * @throws LlamaException if the model cannot be loaded or training fails
     */
    public static void finetune(Path model, String trainingText, Path output, int epochs, float learningRate) {
        finetune(model, trainingText, output, epochs, learningRate, 0, -1);
    }

    /**
     * Fine-tune {@code model} on {@code trainingText}, with explicit context size and GPU offload.
     *
     * @param model the base GGUF model to fine-tune
     * @param trainingText the training corpus (tokenized in-process)
     * @param output the path the fine-tuned GGUF is written to
     * @param epochs number of passes over the corpus (at least 1)
     * @param learningRate the AdamW learning rate at the first epoch (e.g. {@code 1e-5f})
     * @param nCtx context size in tokens, or {@code 0} to use the model's trained context
     * @param nGpuLayers number of layers to offload to the GPU, or {@code -1} for automatic
     * @throws LlamaException if the model cannot be loaded or training fails
     */
    public static void finetune(
            Path model, String trainingText, Path output, int epochs, float learningRate, int nCtx, int nGpuLayers) {
        String error =
                finetuneNative(
                        model.toString(), trainingText, output.toString(), epochs, learningRate, nCtx, nGpuLayers);
        if (error != null && !error.isEmpty()) {
            throw new LlamaException(error);
        }
    }

    private static native String finetuneNative(
            String modelPath,
            String trainingText,
            String outputPath,
            int epochs,
            float learningRate,
            int nCtx,
            int nGpuLayers);
}
