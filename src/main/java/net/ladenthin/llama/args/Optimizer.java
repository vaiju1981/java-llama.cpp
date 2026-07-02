// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * Optimizer used by {@link net.ladenthin.llama.LlamaTrainer} fine-tuning, mapping to llama.cpp's
 * {@code ggml_opt_optimizer_type}.
 */
public enum Optimizer {

    /** Adam with decoupled weight decay ({@code GGML_OPT_OPTIMIZER_TYPE_ADAMW}). The default. */
    ADAMW(0),

    /** Stochastic gradient descent ({@code GGML_OPT_OPTIMIZER_TYPE_SGD}). */
    SGD(1);

    private final int nativeValue;

    Optimizer(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    /**
     * The integer value passed to the native layer (matches the {@code ggml_opt_optimizer_type} enum).
     *
     * @return the native optimizer-type ordinal
     */
    public int getNativeValue() {
        return nativeValue;
    }
}
