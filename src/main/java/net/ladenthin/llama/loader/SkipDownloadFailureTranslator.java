// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import net.ladenthin.llama.args.ModelFlag;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.exception.ModelUnavailableException;
import net.ladenthin.llama.parameters.ModelParameters;

/**
 * Pure-Java translator from the generic {@link net.ladenthin.llama.exception.LlamaException} raised by the JNI
 * loader to the typed {@link net.ladenthin.llama.exception.ModelUnavailableException} when
 * {@link net.ladenthin.llama.args.ModelFlag#SKIP_DOWNLOAD} is set and the load failed because the
 * configured model file was missing or invalid.
 *
 * <p>Lives outside {@link net.ladenthin.llama.LlamaModel} so that unit tests can exercise the
 * translation heuristic without triggering {@code LlamaModel}'s
 * {@link LlamaLoader} static initializer (which loads the JNI library and is
 * not available in CPU-only / non-native test environments).</p>
 *
 * <h2>Why a heuristic and not a direct exception catch</h2>
 *
 * <p>{@code --skip-download} is not a registered upstream argument, so passing
 * it makes upstream arg parsing fail and {@code common_params_parse} return
 * {@code false}. The JNI layer reports that {@code false} return as a generic
 * {@link net.ladenthin.llama.exception.LlamaException} with the message
 * {@value #LOAD_PARSE_FAILED_MESSAGE}. The Java layer recognises the combined
 * signal: {@code SKIP_DOWNLOAD} flag set + JNI message matches. (Earlier
 * llama.cpp builds raised a {@code common_skip_download_exception} inside
 * {@code common_download_file_single} for this case, caught within upstream's own
 * {@code common_params_parse_ex}; that type and the {@code skip_download} option
 * were removed in b9803, but the heuristic is unaffected because it keys on the
 * parse-failure message rather than the C++ exception.)</p>
 */
public final class SkipDownloadFailureTranslator {

    /**
     * Substring used by the JNI bridge when {@code common_params_parse} returns
     * {@code false}; matched at the Java layer to recognise the
     * {@code SKIP_DOWNLOAD} case.
     */
    static final String LOAD_PARSE_FAILED_MESSAGE = "Failed to parse model parameters";

    private SkipDownloadFailureTranslator() {
        // utility — not instantiable
    }

    /**
     * Translates a generic load failure into a typed
     * {@link net.ladenthin.llama.exception.ModelUnavailableException} when the user opted into
     * {@link net.ladenthin.llama.args.ModelFlag#SKIP_DOWNLOAD} and the JNI surfaced the
     * {@value #LOAD_PARSE_FAILED_MESSAGE} message; otherwise returns the
     * original exception unchanged so the caller can re-throw it as-is.
     *
     * @param parameters the parameters passed to the failing constructor
     * @param original   the original load failure to translate or pass through
     * @return a {@link net.ladenthin.llama.exception.ModelUnavailableException} when the heuristic matches;
     *         otherwise the original {@code LlamaException}
     */
    public static LlamaException translate(ModelParameters parameters, LlamaException original) {
        if (parameters.hasFlag(ModelFlag.SKIP_DOWNLOAD)
                && original.getMessage() != null
                && original.getMessage().contains(LOAD_PARSE_FAILED_MESSAGE)) {
            return new ModelUnavailableException(
                    "Model unavailable: --skip-download is set but the configured model file is missing or "
                            + "invalid (no download attempted).",
                    original);
        }
        return original;
    }
}
