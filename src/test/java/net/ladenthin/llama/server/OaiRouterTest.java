// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose =
                "Verify OaiRouter dispatches each OAI endpoint to the backend, forwards the request body, enforces "
                        + "method/body preconditions (405/400), returns 404 for unknown paths, strips query strings, and "
                        + "converts backend exceptions into a 500 OpenAI error envelope. Uses a fake backend (no native model).")
public class OaiRouterTest {

    private static final String CHAT_RESPONSE = "{\"object\":\"chat.completion\"}";
    private static final String COMPLETION_RESPONSE = "{\"object\":\"text_completion\"}";
    private static final String EMBED_RESPONSE = "{\"object\":\"list\",\"data\":[]}";
    private static final String MODELS_RESPONSE = "{\"object\":\"list\",\"data\":[{\"id\":\"m\"}]}";

    /** Records the last forwarded body and returns canned per-endpoint JSON. */
    private static final class RecordingBackend implements OaiBackend {
        private String lastBody = "";

        @Override
        public String chatCompletions(String requestJson) {
            lastBody = requestJson;
            return CHAT_RESPONSE;
        }

        @Override
        public String completions(String requestJson) {
            lastBody = requestJson;
            return COMPLETION_RESPONSE;
        }

        @Override
        public String embeddings(String requestJson) {
            lastBody = requestJson;
            return EMBED_RESPONSE;
        }

        @Override
        public String listModels() {
            return MODELS_RESPONSE;
        }

        String lastBody() {
            return lastBody;
        }
    }

    private static final class ThrowingBackend implements OaiBackend {
        @Override
        public String chatCompletions(String requestJson) {
            throw new IllegalStateException("boom");
        }

        @Override
        public String completions(String requestJson) {
            throw new IllegalStateException("boom");
        }

        @Override
        public String embeddings(String requestJson) {
            throw new IllegalStateException("boom");
        }

        @Override
        public String listModels() {
            throw new IllegalStateException("boom");
        }
    }

    @Test
    public void chatCompletionsForwardsBodyAndReturnsResponse() {
        RecordingBackend backend = new RecordingBackend();
        OaiRouter router = new OaiRouter(backend);
        OaiResponse resp = router.route("POST", "/v1/chat/completions", "{\"messages\":[]}");
        assertThat(resp.getStatus(), is(200));
        assertThat(resp.getBody(), is(CHAT_RESPONSE));
        assertThat(backend.lastBody(), is("{\"messages\":[]}"));
    }

    @Test
    public void completionsRoute() {
        OaiResponse resp =
                new OaiRouter(new RecordingBackend()).route("POST", "/v1/completions", "{\"prompt\":\"hi\"}");
        assertThat(resp.getStatus(), is(200));
        assertThat(resp.getBody(), is(COMPLETION_RESPONSE));
    }

    @Test
    public void embeddingsRoute() {
        OaiResponse resp = new OaiRouter(new RecordingBackend()).route("POST", "/v1/embeddings", "{\"input\":\"hi\"}");
        assertThat(resp.getStatus(), is(200));
        assertThat(resp.getBody(), is(EMBED_RESPONSE));
    }

    @Test
    public void modelsRoute() {
        OaiResponse resp = new OaiRouter(new RecordingBackend()).route("GET", "/v1/models", null);
        assertThat(resp.getStatus(), is(200));
        assertThat(resp.getBody(), is(MODELS_RESPONSE));
    }

    @Test
    public void modelsRouteIgnoresQueryString() {
        OaiResponse resp = new OaiRouter(new RecordingBackend()).route("GET", "/v1/models?limit=1", null);
        assertThat(resp.getStatus(), is(200));
        assertThat(resp.getBody(), is(MODELS_RESPONSE));
    }

    @Test
    public void healthRoutes() {
        OaiRouter router = new OaiRouter(new RecordingBackend());
        assertThat(router.route("GET", "/health", null).getStatus(), is(200));
        assertThat(router.route("GET", "/health", null).getBody(), containsString("\"status\":\"ok\""));
        assertThat(router.route("GET", "/", null).getStatus(), is(200));
    }

    @Test
    public void wrongMethodYields405() {
        OaiRouter router = new OaiRouter(new RecordingBackend());
        assertThat(router.route("GET", "/v1/chat/completions", null).getStatus(), is(405));
        assertThat(router.route("POST", "/v1/models", "{}").getStatus(), is(405));
    }

    @Test
    public void emptyOrNullBodyYields400() {
        OaiRouter router = new OaiRouter(new RecordingBackend());
        assertThat(router.route("POST", "/v1/chat/completions", null).getStatus(), is(400));
        assertThat(router.route("POST", "/v1/chat/completions", "   ").getStatus(), is(400));
    }

    @Test
    public void unknownPathYields404() {
        OaiResponse resp = new OaiRouter(new RecordingBackend()).route("GET", "/v1/nope", null);
        assertThat(resp.getStatus(), is(404));
        assertThat(resp.getBody(), containsString("\"type\":\"not_found\""));
        assertThat(resp.getBody(), containsString("/v1/nope"));
    }

    @Test
    public void backendExceptionYields500() {
        OaiResponse resp = new OaiRouter(new ThrowingBackend()).route("POST", "/v1/chat/completions", "{}");
        assertThat(resp.getStatus(), is(500));
        assertThat(resp.getBody(), containsString("\"type\":\"internal_error\""));
        assertThat(resp.getBody(), containsString("boom"));
    }
}
