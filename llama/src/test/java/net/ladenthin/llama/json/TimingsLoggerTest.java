// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.value.Timings;
import nl.altindag.log.LogCaptor;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Pin the per-run timing-line format (TimingsLogger#format) byte-for-byte "
                + "and verify the SLF4J pipeline on the dedicated 'net.ladenthin.llama.timings' "
                + "logger so a future format regression or accidental log-suppression is caught "
                + "at test time.")
public class TimingsLoggerTest {

    /** Format check on a typical generation (no speculative decoding). */
    @Test
    public void format_standardGeneration_singleLineWithAllSegments() {
        Timings t = new Timings(
                /*cacheN*/ 0,
                /*promptN*/ 12,
                /*promptMs*/ 84.3,
                /*promptPerSec*/ 142.4,
                /*predictedN*/ 256,
                /*predictedMs*/ 5031.7,
                /*predictedPerSec*/ 50.9,
                /*draftN*/ 0,
                /*draftNAccepted*/ 0);

        String line = TimingsLogger.format(t);

        assertEquals(
                "prompt: 12 tok in 84.3 ms (142.4 tok/s)" + " | gen: 256 tok in 5031.7 ms (50.9 tok/s)" + " | cache: 0",
                line);
    }

    /** Speculative-decoding runs append a {@code | draft: N (M accepted)} segment. */
    @Test
    public void format_speculativeDecoding_includesDraftSegment() {
        Timings t = new Timings(0, 4, 10.0, 400.0, 100, 1000.0, 100.0, 50, 35);

        String line = TimingsLogger.format(t);

        assertTrue(line.contains(" | draft: 50 (35 accepted)"), line);
    }

    /** Non-speculative runs do NOT append the draft segment. */
    @Test
    public void format_nonSpeculativeRun_omitsDraftSegment() {
        Timings t = new Timings(0, 4, 10.0, 400.0, 100, 1000.0, 100.0, 0, 0);

        String line = TimingsLogger.format(t);

        assertFalse(line.contains("draft"), line);
    }

    /** Cache-hit count is rendered as-is so users can spot prompt-prefix reuse. */
    @Test
    public void format_cacheHits_renderedExactly() {
        Timings t = new Timings(64, 12, 84.3, 142.4, 256, 5031.7, 50.9, 0, 0);

        String line = TimingsLogger.format(t);

        assertTrue(line.contains(" | cache: 64"), line);
    }

    /**
     * Pipeline check: emit through the dedicated SLF4J logger and assert
     * LogCaptor sees the formatted line at INFO level.
     */
    @Test
    public void log_pipelineDelivery_emitsFormattedLineAtInfo() {
        Timings t = new Timings(0, 12, 84.3, 142.4, 256, 5031.7, 50.9, 0, 0);

        try (LogCaptor captor = LogCaptor.forName(TimingsLogger.LOGGER_NAME)) {
            TimingsLogger.log(t);

            assertEquals(1, captor.getInfoLogs().size());
            assertEquals(TimingsLogger.format(t), captor.getInfoLogs().get(0));
        }
    }

    /** Empty timings (all-zero, typically a parse failure) are not logged. */
    @Test
    public void log_allZeroTimings_skipsEmptyLine() {
        Timings allZero = Timings.fromJson(null);

        try (LogCaptor captor = LogCaptor.forName(TimingsLogger.LOGGER_NAME)) {
            TimingsLogger.log(allZero);

            assertTrue(captor.getInfoLogs().isEmpty(), "expected no log lines for all-zero timings");
        }
    }

    /** Null is treated as a no-op so callers don't need to null-check. */
    @Test
    public void log_nullTimings_isNoOp() {
        try (LogCaptor captor = LogCaptor.forName(TimingsLogger.LOGGER_NAME)) {
            TimingsLogger.log(null);

            assertTrue(captor.getInfoLogs().isEmpty(), "expected no log lines when input is null");
        }
    }
}
