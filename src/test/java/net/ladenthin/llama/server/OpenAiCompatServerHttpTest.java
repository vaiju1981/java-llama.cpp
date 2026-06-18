// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import org.junit.jupiter.api.Test;

/**
 * End-to-end HTTP tests for {@link OpenAiCompatServer} driven over a real socket with a
 * {@link FakeChatBackend} — no native library and no model are loaded. Exercises routing,
 * authentication, the non-streaming and Server-Sent-Events paths, heartbeats, and error statuses.
 *
 * <p>HTTP request plumbing is inherited from {@link OpenAiServerTestSupport}.
 */
public class OpenAiCompatServerHttpTest extends OpenAiServerTestSupport {

    private static final String CHAT_BODY = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";

    private static OpenAiServerConfig config() {
        return OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .modelId("test-model")
                .build();
    }

    @Test
    public void nonStreamingReturnsTheCompletionBody() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeChatBackend(), config()).start()) {
            Response response = post(server.getPort(), "/v1/chat/completions", CHAT_BODY, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("chat.completion"));
            assertThat(response.body, containsString("hello"));
        }
    }

    @Test
    public void streamingReturnsSseChunksThenDone() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeChatBackend(), config()).start()) {
            String body = "{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response response = post(server.getPort(), "/v1/chat/completions", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("data: "));
            assertThat(response.body, containsString("chat.completion.chunk"));
            assertThat(response.body, containsString("data: [DONE]"));
        }
    }

    @Test
    public void streamingEmitsHeartbeatsDuringAGap() throws IOException {
        OpenAiServerConfig cfg = OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .heartbeatMillis(50L)
                .build();
        try (OpenAiCompatServer server = new OpenAiCompatServer(new SlowFakeChatBackend(), cfg).start()) {
            String body = "{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response response = post(server.getPort(), "/v1/chat/completions", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString(":")); // SSE comment heartbeat
            assertThat(response.body, containsString("data: [DONE]"));
        }
    }

    @Test
    public void modelsEndpointAdvertisesConfiguredModel() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeChatBackend(), config()).start()) {
            Response response = get(server.getPort(), "/v1/models", "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("test-model"));
        }
    }

    @Test
    public void unknownPathReturns404() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeChatBackend(), config()).start()) {
            Response response = get(server.getPort(), "/v1/embeddings", "");
            assertThat(response.code, is(404));
        }
    }

    @Test
    public void missingMessagesReturns400() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeChatBackend(), config()).start()) {
            Response response = post(server.getPort(), "/v1/chat/completions", "{}", "");
            assertThat(response.code, is(400));
        }
    }

    @Test
    public void malformedJsonReturns400() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeChatBackend(), config()).start()) {
            Response response = post(server.getPort(), "/v1/chat/completions", "not json", "");
            assertThat(response.code, is(400));
        }
    }

    @Test
    public void getOnChatCompletionsReturns405() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeChatBackend(), config()).start()) {
            Response response = get(server.getPort(), "/v1/chat/completions", "");
            assertThat(response.code, is(405));
        }
    }

    @Test
    public void authRequiredWhenApiKeyConfigured() throws IOException {
        OpenAiServerConfig cfg = OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .apiKey("secret")
                .build();
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeChatBackend(), cfg).start()) {
            int port = server.getPort();
            assertThat(post(port, "/v1/chat/completions", CHAT_BODY, "").code, is(401));
            assertThat(post(port, "/v1/chat/completions", CHAT_BODY, "Bearer wrong").code, is(401));
            assertThat(post(port, "/v1/chat/completions", CHAT_BODY, "Bearer secret").code, is(200));
        }
    }

    /** Deterministic backend that returns canned OpenAI shapes. */
    static final class FakeChatBackend implements ChatBackend {
        @Override
        public String complete(JsonNode request) {
            return "{\"object\":\"chat.completion\",\"choices\":[{\"index\":0,"
                    + "\"message\":{\"role\":\"assistant\",\"content\":\"hello\"}}]}";
        }

        @Override
        public void stream(JsonNode request, ChunkSink sink) throws IOException {
            sink.accept("{\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"he\"}}]}");
            sink.accept("{\"object\":\"chat.completion.chunk\","
                    + "\"choices\":[{\"delta\":{\"content\":\"llo\"},\"finish_reason\":\"stop\"}]}");
        }
    }

    /** Backend that stalls before emitting, so the server's heartbeat fires during the gap. */
    static final class SlowFakeChatBackend implements ChatBackend {
        @Override
        public String complete(JsonNode request) {
            return "{\"object\":\"chat.completion\",\"choices\":[]}";
        }

        @Override
        public void stream(JsonNode request, ChunkSink sink) throws IOException {
            try {
                Thread.sleep(300L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
            sink.accept("{\"object\":\"chat.completion.chunk\","
                    + "\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}");
        }
    }
}
