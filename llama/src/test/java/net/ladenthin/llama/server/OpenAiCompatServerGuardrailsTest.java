// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import org.junit.jupiter.api.Test;

/**
 * Model-free coverage of the M5 multi-client guardrails: the token-bucket rate limit (HTTP 429 once
 * the per-client budget is exhausted) and runtime API-key rotation via {@link OpenAiCompatServer#setApiKey}.
 * A {@link FakeM1Backend} returns canned JSON for the new {@link OpenAiBackend} default methods, so no
 * native library or model is required.
 */
public class OpenAiCompatServerGuardrailsTest extends OpenAiServerTestSupport {

    private OpenAiCompatServer startServer(OpenAiServerConfig config) throws Exception {
        return new OpenAiCompatServer(new FakeM1Backend(), config).start();
    }

    @Test
    public void rateLimitReturns429OnceBudgetExhausted() throws Exception {
        // 1 request/sec, burst capacity 1: the first call passes, the rest are throttled.
        OpenAiServerConfig config =
                OpenAiServerConfig.builder().port(0).rateLimitRps(1.0).build();
        try (OpenAiCompatServer server = startServer(config)) {
            assertThat(get(server.getPort(), "/health", "").code, is(200));
            assertThat(get(server.getPort(), "/health", "").code, is(429));
            assertThat(get(server.getPort(), "/health", "").code, is(429));
        }
    }

    @Test
    public void apiKeyRotationUpdatesAcceptedKey() throws Exception {
        OpenAiServerConfig config =
                OpenAiServerConfig.builder().port(0).apiKey("first-key").build();
        try (OpenAiCompatServer server = startServer(config)) {
            assertThat(get(server.getPort(), "/models", "Bearer first-key").code, is(200));
            assertThat(get(server.getPort(), "/models", "Bearer second-key").code, is(401));

            server.setApiKey("second-key");

            assertThat(get(server.getPort(), "/models", "Bearer first-key").code, is(401));
            assertThat(get(server.getPort(), "/models", "Bearer second-key").code, is(200));
        }
    }

    /** Minimal backend: only the routes exercised by these tests need real bodies. */
    static final class FakeM1Backend implements OpenAiBackend {
        @Override
        public String complete(com.fasterxml.jackson.databind.JsonNode request) {
            return "{}";
        }

        @Override
        public void stream(com.fasterxml.jackson.databind.JsonNode request, ChunkSink sink) {}

        @Override
        public String completions(com.fasterxml.jackson.databind.JsonNode request) {
            return "{}";
        }

        @Override
        public String embeddings(com.fasterxml.jackson.databind.JsonNode request) {
            return "{}";
        }

        @Override
        public String rerank(com.fasterxml.jackson.databind.JsonNode request) {
            return "{}";
        }

        @Override
        public String infill(com.fasterxml.jackson.databind.JsonNode request) {
            return "{}";
        }
    }
}
