// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.ladenthin.llama.args.ModelFlag;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for the {@code SKIP_DOWNLOAD} plumbing on {@link ModelParameters} and the
 * paired translation in {@link SkipDownloadFailureTranslator}.
 *
 * <p>These tests do NOT load the native library &#x2014; they exercise pure Java logic:
 * the boolean-setter round-trip via {@link ModelParameters#hasFlag(ModelFlag)} and the
 * static translation heuristic that promotes a generic {@link LlamaException} to a typed
 * {@link ModelUnavailableException} when the {@link ModelFlag#SKIP_DOWNLOAD} flag is
 * set.</p>
 */
public class LlamaModelSkipDownloadTest {

    /** Default constructor used by JUnit Jupiter. */
    public LlamaModelSkipDownloadTest() {
        // no-op
    }

    @Test
    @DisplayName("setSkipDownload(true) sets the SKIP_DOWNLOAD flag")
    public void setSkipDownload_true_setsFlag() {
        ModelParameters p = new ModelParameters().setSkipDownload(true);
        assertTrue(p.hasFlag(ModelFlag.SKIP_DOWNLOAD));
    }

    @Test
    @DisplayName("setSkipDownload(false) clears the SKIP_DOWNLOAD flag")
    public void setSkipDownload_false_clearsFlag() {
        ModelParameters p = new ModelParameters().setSkipDownload(true).setSkipDownload(false);
        assertFalse(p.hasFlag(ModelFlag.SKIP_DOWNLOAD));
    }

    @Test
    @DisplayName("hasFlag returns false by default")
    public void hasFlag_byDefault_returnsFalse() {
        assertFalse(new ModelParameters().hasFlag(ModelFlag.SKIP_DOWNLOAD));
    }

    @Test
    @DisplayName("translate: SKIP_DOWNLOAD set + 'Failed to parse' message -> ModelUnavailableException")
    public void translate_skipDownloadSetAndParseFailed_returnsTypedException() {
        ModelParameters p = new ModelParameters().setSkipDownload(true);
        LlamaException original = new LlamaException("Failed to parse model parameters");

        LlamaException translated = SkipDownloadFailureTranslator.translate(p, original);

        assertInstanceOf(ModelUnavailableException.class, translated);
        assertNotNull(translated.getMessage());
        assertTrue(
                translated.getMessage().contains("--skip-download"),
                "message should mention the --skip-download flag for caller diagnosis");
        assertSame(original, translated.getCause(), "original exception should be preserved as cause");
    }

    @Test
    @DisplayName("translate: SKIP_DOWNLOAD set but unrelated message -> original exception passes through")
    public void translate_skipDownloadSetButUnrelatedMessage_returnsOriginal() {
        ModelParameters p = new ModelParameters().setSkipDownload(true);
        LlamaException original = new LlamaException("could not allocate VRAM");

        LlamaException translated = SkipDownloadFailureTranslator.translate(p, original);

        assertSame(original, translated);
    }

    @Test
    @DisplayName("translate: SKIP_DOWNLOAD NOT set -> original exception passes through even on parse-failed")
    public void translate_skipDownloadNotSet_returnsOriginal() {
        ModelParameters p = new ModelParameters(); // skip-download not set
        LlamaException original = new LlamaException("Failed to parse model parameters");

        LlamaException translated = SkipDownloadFailureTranslator.translate(p, original);

        assertSame(original, translated);
    }

    @Test
    @DisplayName("translate: null message -> original exception passes through")
    public void translate_nullMessage_returnsOriginal() {
        ModelParameters p = new ModelParameters().setSkipDownload(true);
        LlamaException original = new LlamaException((String) null);

        LlamaException translated = SkipDownloadFailureTranslator.translate(p, original);

        assertSame(original, translated);
    }
}
