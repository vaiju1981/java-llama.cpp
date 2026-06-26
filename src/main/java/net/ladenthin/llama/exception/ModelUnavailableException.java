// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.exception;

import net.ladenthin.llama.parameters.ModelParameters;

/**
 * Thrown by {@link net.ladenthin.llama.LlamaModel#LlamaModel(ModelParameters)} when
 * {@link net.ladenthin.llama.args.ModelFlag#OFFLINE} (or {@link net.ladenthin.llama.parameters.ModelParameters#setOffline(boolean)
 * setOffline(true)}) is set and the configured local model file does not exist
 * &#x2014; i.e. the loader would have had to download a replacement but is
 * forbidden to.
 *
 * <p>Lets air-gapped / pre-staged-model deployments distinguish &quot;model file
 * absent&quot; from generic configuration errors. The check is a deterministic
 * pre-check in {@link net.ladenthin.llama.loader.OfflineModelGuard}: when
 * {@code --offline} is set and the configured local {@code --model} path points at
 * a file that does not exist, the loader throws this typed exception before calling
 * the native loader, rather than letting it surface as a generic load failure.
 * (The predecessor used a parse-failure heuristic for the removed
 * {@code --skip-download} flag; that flag and the {@code common_skip_download_exception}
 * it referenced were removed in llama.cpp b9803 and replaced here by {@code --offline}.)</p>
 */
public class ModelUnavailableException extends LlamaException {

    /**
     * Creates a new {@link ModelUnavailableException} with the given message.
     *
     * @param message the detail message; may be {@code null}
     */
    public ModelUnavailableException(String message) {
        super(message);
    }

    /**
     * Creates a new {@link ModelUnavailableException} with the given message and cause.
     *
     * @param message the detail message; may be {@code null}
     * @param cause   the underlying cause; may be {@code null}
     */
    public ModelUnavailableException(String message, Throwable cause) {
        super(message, cause);
    }
}
