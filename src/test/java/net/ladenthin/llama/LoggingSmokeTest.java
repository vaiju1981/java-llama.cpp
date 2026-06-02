// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import nl.altindag.log.LogCaptor;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

@ClaudeGenerated(
        purpose = "Smoke-test the SLF4J + Logback pipeline so a future binding or "
                + "configuration regression is caught at test time rather than silently "
                + "swallowing logs in production.")
public class LoggingSmokeTest {

    /**
     * Direct binding/routing check: emit a known event through the configured
     * pipeline and assert LogCaptor saw it. Fails if SLF4J binds to NOPLogger
     * or if Logback is misconfigured to drop INFO from this logger.
     */
    @Test
    public void slf4jPipelineEmits() {
        try (LogCaptor captor = LogCaptor.forClass(OSInfo.class)) {
            LoggerFactory.getLogger(OSInfo.class).info("smoke");
            assertTrue(
                    captor.getInfoLogs().contains("smoke"),
                    "SLF4J pipeline did not deliver INFO event to LogCaptor; "
                            + "binding or Logback config is broken");
        }
    }

    /**
     * Production call-site check: trigger {@link OSInfo#getHardwareName()} on a
     * stub {@link ProcessRunner} that throws, and assert the catch-block's
     * {@code error} log is captured. Pins the production log line as part of
     * the contract — an accidental refactor that drops the logger call fails
     * this test.
     */
    @Test
    public void getHardwareNameLogsError_whenProcessRunnerThrows() {
        ProcessRunner original = OSInfo.processRunner;
        try (LogCaptor captor = LogCaptor.forClass(OSInfo.class)) {
            OSInfo.processRunner = new ProcessRunner() {
                @Override
                String runAndWaitFor(String command) throws IOException {
                    throw new IOException("boom");
                }
            };
            assertEquals("unknown", OSInfo.getHardwareName());
            assertTrue(
                    captor.getErrorLogs().stream()
                            .anyMatch(m -> m.contains("Error while running uname -m")),
                    "expected error log 'Error while running uname -m' was not captured");
        } finally {
            OSInfo.processRunner = original;
        }
    }
}
