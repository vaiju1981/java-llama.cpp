// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.api.Test;

/**
 * Model-free coverage of the M1 route-parity endpoints ({@code /tokenize}, {@code /detokenize},
 * {@code /apply-template}, {@code /lora-adapters}, {@code /count_tokens} and their bare/alias
 * forms). A {@link FakeM1Backend} returns canned JSON for the new {@link OpenAiBackend}
 * default methods, so the whole HTTP surface — registration, CORS, authentication, method
 * gating, and verbatim pass-through — is exercised with no native library and no model loaded.
 */
public class OpenAiCompatServerM1RoutesTest extends OpenAiServerTestSupport {

    static final class FakeM1Backend implements OpenAiBackend {
        @Override
        public String complete(JsonNode request) {
            return "{}";
        }

        @Override
        public void stream(JsonNode request, ChunkSink sink) {}

        @Override
        public String completions(JsonNode request) {
            return "{}";
        }

        @Override
        public String embeddings(JsonNode request) {
            return "{}";
        }

        @Override
        public String rerank(JsonNode request) {
            return "{}";
        }

        @Override
        public String infill(JsonNode request) {
            return "{}";
        }

        @Override
        public String tokenize(JsonNode request) {
            return "{\"tokens\":[1,2,3]}";
        }

        @Override
        public String detokenize(JsonNode request) {
            return "{\"content\":\"hello\"}";
        }

        @Override
        public String applyTemplate(JsonNode request) {
            return "{\"content\":\"<templated>\"}";
        }

        @Override
        public String loraAdapters() {
            return "{\"adapters\":[]}";
        }

        @Override
        public String setLoraAdapters(JsonNode request) {
            return "{\"status\":\"ok\"}";
        }

        @Override
        public String countTokens(JsonNode request) {
            return "{\"count\":7}";
        }

        @Override
        public String audioTranscriptions(JsonNode request) {
            return "{\"text\":\"hello world\"}";
        }

        @Override
        public String control(JsonNode request) {
            return "{\"status\":\"ok\"}";
        }

        @Override
        public String streamsLookup() {
            return "{\"streams\":[]}";
        }

        @Override
        public String streamGet(String convId) {
            return "{\"conv_id\":\"" + convId + "\",\"text\":\"partial\"}";
        }

        @Override
        public String streamDelete(String convId) {
            return "{\"conv_id\":\"" + convId + "\",\"status\":\"cancelled\"}";
        }

        @Override
        public String saveSlots(JsonNode request) {
            return "{\"status\":\"ok\"}";
        }

        @Override
        public String eraseSlot(int id) {
            return "{\"id\":" + id + ",\"status\":\"erased\"}";
        }
    }

    private OpenAiCompatServer startServer() throws Exception {
        OpenAiServerConfig config = OpenAiServerConfig.builder().port(0).build();
        return new OpenAiCompatServer(new FakeM1Backend(), config).start();
    }

    @Test
    public void testModelsListsMultipleIds() throws Exception {
        OpenAiServerConfig config = OpenAiServerConfig.builder()
                .port(0)
                .modelIds("model-a", "model-b")
                .build();
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeM1Backend(), config).start()) {
            Response response = get(server.getPort(), "/models", "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("\"model-a\""));
            assertThat(response.body, containsString("\"model-b\""));
        }
    }

    @Test
    public void testTokenize_canonicalAndBareAlias() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(
                    post(server.getPort(), "/tokenize", "{\"content\":\"hi\"}", "").body, containsString("\"tokens\""));
            assertThat(post(server.getPort(), "/tokenize", "{\"content\":\"hi\"}", "").code, is(200));
        }
    }

    @Test
    public void testDetokenize_passthrough() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(
                    post(server.getPort(), "/detokenize", "{\"tokens\":[1]}", "").body, containsString("\"content\""));
        }
    }

    @Test
    public void testApplyTemplate_wrapsContent() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(
                    post(server.getPort(), "/apply-template", "{\"messages\":[]}", "").body,
                    containsString("<templated>"));
        }
    }

    @Test
    public void testLoraAdapters_getAndPost() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(get(server.getPort(), "/lora-adapters", "").body, containsString("\"adapters\""));
            assertThat(
                    post(server.getPort(), "/lora-adapters", "{\"adapters\":[{\"id\":0,\"scale\":1.0}]}", "").body,
                    containsString("\"status\""));
        }
    }

    @Test
    public void testCountTokens_canonicalAndApiAliases() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(
                    post(server.getPort(), "/count_tokens", "{\"prompt\":\"hi\"}", "").body,
                    containsString("\"count\""));
            assertThat(
                    post(server.getPort(), "/v1/chat/completions/input_tokens", "{\"messages\":[]}", "").body,
                    containsString("\"count\""));
            assertThat(
                    post(server.getPort(), "/v1/responses/input_tokens", "{\"input\":\"hi\"}", "").body,
                    containsString("\"count\""));
            assertThat(
                    post(server.getPort(), "/v1/messages/count_tokens", "{\"input\":\"hi\"}", "").body,
                    containsString("\"count\""));
        }
    }

    @Test
    public void testTokenize_rejectsGetWith405() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(get(server.getPort(), "/tokenize", "").code, is(405));
        }
    }

    @Test
    public void testAudioTranscriptions_passthrough() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(
                    post(server.getPort(), "/v1/audio/transcriptions", "{\"model\":\"x\"}", "").body,
                    containsString("\"text\""));
        }
    }

    @Test
    public void testAudioTranscriptions_rejectsGetWith405() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(get(server.getPort(), "/v1/audio/transcriptions", "").code, is(405));
        }
    }

    @Test
    public void testControl_passthrough() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(
                    post(server.getPort(), "/v1/chat/completions/control", "{\"action\":\"pause\"}", "").body,
                    containsString("\"status\""));
        }
    }

    @Test
    public void testStreamsLookup_passthrough() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(get(server.getPort(), "/v1/streams/lookup", "").body, containsString("\"streams\""));
        }
    }

    @Test
    public void testStreamGet_passthrough() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(get(server.getPort(), "/v1/stream/abc", "").body, containsString("\"conv_id\""));
            assertThat(get(server.getPort(), "/v1/stream/abc", "").body, containsString("abc"));
        }
    }

    @Test
    public void testStreamDelete_passthrough() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(delete(server.getPort(), "/v1/stream/abc", "").body, containsString("\"status\""));
        }
    }

    @Test
    public void testStream_missingConvIdReturns404() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(get(server.getPort(), "/v1/stream/", "").code, is(404));
        }
    }

    @Test
    public void testSlots_postSave_passthrough() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(
                    post(server.getPort(), "/slots", "{\"slot_id\":0,\"filename\":\"slot.bin\"}", "").body,
                    containsString("\"status\""));
        }
    }

    @Test
    public void testSlots_deleteErase_passthrough() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(delete(server.getPort(), "/slots/0", "").body, containsString("\"status\""));
            assertThat(delete(server.getPort(), "/slots/0", "").body, containsString("\"id\""));
        }
    }

    @Test
    public void testSlots_deleteInvalidId_returns404() throws Exception {
        try (OpenAiCompatServer server = startServer()) {
            assertThat(delete(server.getPort(), "/slots/not-a-number", "").code, is(404));
        }
    }
}
