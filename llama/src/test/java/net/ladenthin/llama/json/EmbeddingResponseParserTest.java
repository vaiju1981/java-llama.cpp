// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link EmbeddingResponseParser}.
 * No native library or model file needed — JSON string literals only.
 */
@ClaudeGenerated(
        purpose = "Verify EmbeddingResponseParser parses the OAI embeddings response shape "
                + "(index-ordered, skipping non-array embeddings) and builds the batch "
                + "{\"input\": [...]} request with correct JSON escaping.")
public class EmbeddingResponseParserTest {

    private final EmbeddingResponseParser parser = new EmbeddingResponseParser();

    // ------------------------------------------------------------------
    // parse(String)
    // ------------------------------------------------------------------

    @Test
    public void testParse_singleEmbedding() {
        String json = "{\"object\":\"list\",\"data\":["
                + "{\"object\":\"embedding\",\"embedding\":[0.1,0.2,0.3],\"index\":0}]}";
        List<float[]> result = parser.parse(json);
        assertThat(result, hasSize(1));
        assertArrayEquals(new float[] {0.1f, 0.2f, 0.3f}, result.get(0), 0.0001f);
    }

    @Test
    public void testParse_orderedByIndexField() {
        // The native scheduler may complete prompts out of order; the parser must
        // restore the request order via each entry's index field.
        String json = "{\"data\":["
                + "{\"embedding\":[2.0],\"index\":1},"
                + "{\"embedding\":[3.0],\"index\":2},"
                + "{\"embedding\":[1.0],\"index\":0}]}";
        List<float[]> result = parser.parse(json);
        assertThat(result, hasSize(3));
        assertArrayEquals(new float[] {1.0f}, result.get(0), 0.0f);
        assertArrayEquals(new float[] {2.0f}, result.get(1), 0.0f);
        assertArrayEquals(new float[] {3.0f}, result.get(2), 0.0f);
    }

    @Test
    public void testParse_missingIndexFallsBackToPosition() {
        String json = "{\"data\":[{\"embedding\":[1.0]},{\"embedding\":[2.0]}]}";
        List<float[]> result = parser.parse(json);
        assertThat(result, hasSize(2));
        assertArrayEquals(new float[] {1.0f}, result.get(0), 0.0f);
        assertArrayEquals(new float[] {2.0f}, result.get(1), 0.0f);
    }

    @Test
    public void testParse_nonArrayEmbeddingSkipped() {
        // encoding_format=base64 renders the embedding as a string; the float parser skips it.
        String json = "{\"data\":[{\"embedding\":\"AAAA\",\"index\":0},{\"embedding\":[1.0],\"index\":1}]}";
        List<float[]> result = parser.parse(json);
        assertThat(result, hasSize(1));
        assertArrayEquals(new float[] {1.0f}, result.get(0), 0.0f);
    }

    @Test
    public void testParse_emptyData() {
        assertThat(parser.parse("{\"data\":[]}"), is(empty()));
    }

    @Test
    public void testParse_missingData() {
        assertThat(parser.parse("{\"object\":\"list\"}"), is(empty()));
    }

    @Test
    public void testParse_malformedJson() {
        assertThat(parser.parse("not json"), is(empty()));
    }

    // ------------------------------------------------------------------
    // toBatchRequestJson(Collection)
    // ------------------------------------------------------------------

    @Test
    public void testToBatchRequestJson_simple() {
        String json = parser.toBatchRequestJson(Arrays.asList("first", "second"));
        assertThat(json, is("{\"input\":[\"first\",\"second\"]}"));
    }

    @Test
    public void testToBatchRequestJson_escapesSpecialCharacters() {
        String json = parser.toBatchRequestJson(Collections.singletonList("say \"hi\"\nplease"));
        assertThat(json, is("{\"input\":[\"say \\\"hi\\\"\\nplease\"]}"));
    }

    @Test
    public void testToBatchRequestJson_keepsUnicodeText() throws java.io.IOException {
        String prompt = "emoji 😀 and umlaut ä";
        String json = parser.toBatchRequestJson(Collections.singletonList(prompt));
        // Round-trip through the shared mapper to prove lossless encoding.
        com.fasterxml.jackson.databind.JsonNode root = EmbeddingResponseParser.OBJECT_MAPPER.readTree(json);
        assertThat(root.path("input").get(0).asText(), is(prompt));
    }

    @Test
    public void testToBatchRequestJson_emptyCollection() {
        assertThat(parser.toBatchRequestJson(Collections.<String>emptyList()), is("{\"input\":[]}"));
    }
}
