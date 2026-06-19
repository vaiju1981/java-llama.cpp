// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import org.junit.jupiter.api.Test;

/** Unit tests for {@link OpenAiSseFormatter}. Pure string/JSON formatting. */
public class OpenAiSseFormatterTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    public void sseDataFramesWithTrailingBlankLine() {
        assertThat(OpenAiSseFormatter.sseData("{\"a\":1}"), is("data: {\"a\":1}\n\n"));
    }

    @Test
    public void sseDoneIsTheOpenAiTerminator() {
        assertThat(OpenAiSseFormatter.sseDone(), is("data: [DONE]\n\n"));
    }

    @Test
    public void heartbeatIsAnSseCommentLine() {
        String hb = OpenAiSseFormatter.heartbeat();
        assertThat(hb.startsWith(":"), is(true));
        assertThat(hb.endsWith("\n\n"), is(true));
    }

    @Test
    public void errorJsonHasOpenAiEnvelopeShape() throws IOException {
        JsonNode error = MAPPER.readTree(OpenAiSseFormatter.errorJson("boom", "server_error", null))
                .path("error");
        assertThat(error.path("message").asText(), is("boom"));
        assertThat(error.path("type").asText(), is("server_error"));
        assertThat(error.path("code").isNull(), is(true));
    }

    @Test
    public void errorJsonIncludesCodeWhenProvided() throws IOException {
        JsonNode error = MAPPER.readTree(OpenAiSseFormatter.errorJson("bad", "invalid_request_error", "E42"))
                .path("error");
        assertThat(error.path("code").asText(), is("E42"));
    }

    @Test
    public void ensureUsageCachedTokens_injectsWhenDetailsMissing() throws IOException {
        String chunk = "{\"object\":\"chat.completion.chunk\",\"choices\":[],"
                + "\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}";
        JsonNode out = MAPPER.readTree(OpenAiSseFormatter.ensureUsageCachedTokens(chunk));
        assertThat(
                out.path("usage")
                        .path("prompt_tokens_details")
                        .path("cached_tokens")
                        .asInt(),
                is(0));
    }

    @Test
    public void ensureUsageCachedTokens_injectsWhenDetailsPresentButNoCachedTokens() throws IOException {
        String chunk = "{\"usage\":{\"prompt_tokens\":5,\"prompt_tokens_details\":{\"audio_tokens\":0}}}";
        JsonNode out = MAPPER.readTree(OpenAiSseFormatter.ensureUsageCachedTokens(chunk));
        JsonNode details = out.path("usage").path("prompt_tokens_details");
        assertThat(details.path("cached_tokens").asInt(), is(0));
        assertThat(details.path("audio_tokens").asInt(), is(0)); // pre-existing detail preserved
    }

    @Test
    public void ensureUsageCachedTokens_leavesAlreadyCorrectChunkUnchanged() {
        String chunk = "{\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":4}}}";
        assertThat(OpenAiSseFormatter.ensureUsageCachedTokens(chunk), is(chunk));
    }

    @Test
    public void ensureUsageCachedTokens_passesThroughDeltaChunkWithNullUsage() {
        String chunk = "{\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"usage\":null}";
        assertThat(OpenAiSseFormatter.ensureUsageCachedTokens(chunk), is(chunk));
    }

    @Test
    public void ensureUsageCachedTokens_passesThroughChunkWithoutUsage() {
        String chunk = "{\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";
        assertThat(OpenAiSseFormatter.ensureUsageCachedTokens(chunk), is(chunk));
    }

    @Test
    public void ensureUsageCachedTokens_malformedUsageChunkReturnedUnchanged() {
        // Contains a quoted "usage" (so it passes the fast path) but is not parseable — must not throw.
        String chunk = "{\"usage\":{ broken";
        assertThat(OpenAiSseFormatter.ensureUsageCachedTokens(chunk), is(chunk));
    }

    @Test
    public void modelsJsonAdvertisesTheConfiguredModel() throws IOException {
        JsonNode root = MAPPER.readTree(OpenAiSseFormatter.modelsJson("gemma-local"));
        assertThat(root.path("object").asText(), is("list"));
        assertThat(root.path("data").get(0).path("id").asText(), is("gemma-local"));
        assertThat(root.path("data").get(0).path("object").asText(), is("model"));
    }
}
