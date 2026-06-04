// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama;

import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Emits a single-line per-run timing summary to the SLF4J logger
 * {@value #LOGGER_NAME}, mirroring what the {@code llama.cpp} command-line tool
 * prints at the end of a generation.
 *
 * <p>Format:</p>
 * <pre>
 * prompt: 12 tok in 84.3 ms (142.4 tok/s) | gen: 256 tok in 5031.7 ms (50.9 tok/s) | cache: 0
 * </pre>
 *
 * <p>Speculative-decoding runs append a {@code | draft: N (M accepted)} segment.
 * Empty {@link Timings} (both {@code promptN} and {@code predictedN} zero) are
 * skipped &mdash; logging the all-zero fallback on a parse failure or on early
 * cancellation is pure noise.</p>
 *
 * <p>The dedicated logger name lets users suppress just this per-run line in
 * logback without touching the rest of the {@code net.ladenthin.llama} logging
 * tree, e.g.:</p>
 * <pre>
 * &lt;logger name=&quot;net.ladenthin.llama.timings&quot; level=&quot;OFF&quot;/&gt;
 * </pre>
 */
public final class TimingsLogger {

    /** Dedicated SLF4J logger name for the per-run timing line. */
    public static final String LOGGER_NAME = "net.ladenthin.llama.timings";

    private static final Logger LOGGER = LoggerFactory.getLogger(LOGGER_NAME);

    private TimingsLogger() {
        // utility class; not instantiable.
    }

    /**
     * Formats a single-line timing summary suitable for the {@value #LOGGER_NAME}
     * SLF4J logger. Exposed for callers that want to emit the same line through
     * a different sink (e.g. {@code System.err} in a CLI tool).
     *
     * @param t the timings to format
     * @return a single-line summary (no trailing newline)
     */
    public static String format(Timings t) {
        StringBuilder sb = new StringBuilder()
                .append("prompt: ").append(t.getPromptN()).append(" tok in ")
                .append(formatMs(t.getPromptMs())).append(" ms (")
                .append(formatRate(t.getPromptPerSecond())).append(" tok/s)")
                .append(" | gen: ").append(t.getPredictedN()).append(" tok in ")
                .append(formatMs(t.getPredictedMs())).append(" ms (")
                .append(formatRate(t.getPredictedPerSecond())).append(" tok/s)")
                .append(" | cache: ").append(t.getCacheN());
        if (t.getDraftN() > 0) {
            sb.append(" | draft: ").append(t.getDraftN())
                    .append(" (").append(t.getDraftNAccepted()).append(" accepted)");
        }
        return sb.toString();
    }

    /**
     * Logs the per-run timing summary at {@code INFO} level on the dedicated
     * {@value #LOGGER_NAME} logger.
     *
     * <p>No-op when the timings carry no useful data (both prompt and predicted
     * token counts are zero &mdash; typically a parse failure or an early
     * cancellation) or when the logger is below {@code INFO}.</p>
     *
     * @param t the timings to log; may be {@code null} (no-op)
     */
    public static void log(Timings t) {
        if (t == null) {
            return;
        }
        if (t.getPromptN() == 0 && t.getPredictedN() == 0) {
            return;
        }
        if (LOGGER.isInfoEnabled()) {
            LOGGER.info(format(t));
        }
    }

    private static String formatMs(double ms) {
        return String.format(Locale.ROOT, "%.1f", ms);
    }

    private static String formatRate(double rate) {
        return String.format(Locale.ROOT, "%.1f", rate);
    }
}
