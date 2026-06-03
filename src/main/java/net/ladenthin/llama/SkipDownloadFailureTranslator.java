// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import net.ladenthin.llama.args.ModelFlag;

/**
 * Pure-Java translator from the generic {@link LlamaException} raised by the JNI
 * loader to the typed {@link ModelUnavailableException} when
 * {@link ModelFlag#SKIP_DOWNLOAD} is set and the load failed because the
 * configured model file was missing or invalid.
 *
 * <p>Lives outside {@link LlamaModel} so that unit tests can exercise the
 * translation heuristic without triggering {@code LlamaModel}'s
 * {@link LlamaLoader} static initializer (which loads the JNI library and is
 * not available in CPU-only / non-native test environments).</p>
 *
 * <h2>Why a heuristic and not a direct exception catch</h2>
 *
 * <p>Upstream raises {@code common_skip_download_exception} inside
 * {@code common_download_file_single} when {@code --skip-download} is set and
 * the file is missing or has a stale ETag. However that exception is caught
 * INSIDE upstream's own {@code common_params_parse_ex} (at
 * {@code common/arg.cpp:476}) and surfaces only as a {@code false} return
 * from {@code common_params_parse}. The JNI layer reports the {@code false}
 * return as a generic {@link LlamaException} with the message
 * {@value #LOAD_PARSE_FAILED_MESSAGE}. The Java layer therefore cannot catch
 * the C++ exception directly and instead recognises the combined signal:
 * {@code SKIP_DOWNLOAD} flag set + JNI message matches.</p>
 */
final class SkipDownloadFailureTranslator {

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
     * {@link ModelUnavailableException} when the user opted into
     * {@link ModelFlag#SKIP_DOWNLOAD} and the JNI surfaced the
     * {@value #LOAD_PARSE_FAILED_MESSAGE} message; otherwise returns the
     * original exception unchanged so the caller can re-throw it as-is.
     *
     * @param parameters the parameters passed to the failing constructor
     * @param original   the original load failure to translate or pass through
     * @return a {@link ModelUnavailableException} when the heuristic matches;
     *         otherwise the original {@code LlamaException}
     */
    static LlamaException translate(ModelParameters parameters, LlamaException original) {
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
