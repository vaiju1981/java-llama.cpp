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
    public void modelsJsonAdvertisesTheConfiguredModel() throws IOException {
        JsonNode root = MAPPER.readTree(OpenAiSseFormatter.modelsJson("gemma-local"));
        assertThat(root.path("object").asText(), is("list"));
        assertThat(root.path("data").get(0).path("id").asText(), is("gemma-local"));
        assertThat(root.path("data").get(0).path("object").asText(), is("model"));
    }
}
