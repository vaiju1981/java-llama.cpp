// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.exception;

import net.ladenthin.llama.parameters.ModelParameters;

/**
 * Thrown by {@link net.ladenthin.llama.LlamaModel#LlamaModel(ModelParameters)} when
 * {@link net.ladenthin.llama.args.ModelFlag#SKIP_DOWNLOAD} (or {@link net.ladenthin.llama.parameters.ModelParameters#setSkipDownload(boolean)
 * setSkipDownload(true)}) is set and the configured model file is missing or
 * invalid &#x2014; i.e. the loader would have had to download a replacement but is
 * forbidden to.
 *
 * <p>Lets air-gapped / pre-staged-model deployments distinguish &quot;model file
 * absent&quot; from generic configuration errors. The {@code --skip-download}
 * flag is not a registered upstream argument, so upstream arg parsing fails and
 * {@code common_params_parse} returns {@code false}; the Java layer combines that
 * parse-failure signal with the {@code SKIP_DOWNLOAD} flag to recognise the
 * skip-download case and translate it to this typed exception. (Earlier llama.cpp
 * builds raised a {@code common_skip_download_exception} for this; that type was
 * removed in b9803 together with the {@code skip_download} download option, but
 * the Java-side heuristic is unaffected because it keys on the parse-failure
 * message, not the C++ exception.)</p>
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
