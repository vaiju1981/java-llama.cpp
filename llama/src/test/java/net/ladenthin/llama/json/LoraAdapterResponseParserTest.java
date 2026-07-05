// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.value.LoraAdapter;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link LoraAdapterResponseParser}.
 * No native library or model file needed — JSON string literals only.
 */
@ClaudeGenerated(
        purpose = "Verify LoraAdapterResponseParser parses the native GET /lora-adapters array "
                + "shape (incl. field defaults and tolerated alora extras) and builds the "
                + "POST /lora-adapters request body with finite-scale validation.")
public class LoraAdapterResponseParserTest {

    private final LoraAdapterResponseParser parser = new LoraAdapterResponseParser();

    // ------------------------------------------------------------------
    // parse(String)
    // ------------------------------------------------------------------

    @Test
    public void testParse_fullEntry() {
        String json = "[{\"id\":0,\"path\":\"adapter.gguf\",\"scale\":0.5,"
                + "\"task_name\":\"classification\",\"prompt_prefix\":\"prefix\"}]";
        List<LoraAdapter> result = parser.parse(json);
        assertThat(result, hasSize(1));
        LoraAdapter adapter = result.get(0);
        assertThat(adapter.getId(), is(0));
        assertThat(adapter.getPath(), is("adapter.gguf"));
        assertEquals(0.5f, adapter.getScale(), 0.0001f);
        assertThat(adapter.getTaskName(), is("classification"));
        assertThat(adapter.getPromptPrefix(), is("prefix"));
    }

    @Test
    public void testParse_missingFieldsFallBackToDefaults() {
        List<LoraAdapter> result = parser.parse("[{}]");
        assertThat(result, hasSize(1));
        LoraAdapter adapter = result.get(0);
        assertThat(adapter.getId(), is(-1));
        assertThat(adapter.getPath(), is(""));
        assertEquals(0.0f, adapter.getScale(), 0.0f);
        assertThat(adapter.getTaskName(), is(""));
        assertThat(adapter.getPromptPrefix(), is(""));
    }

    @Test
    public void testParse_aloraExtrasAreTolerated() {
        // aLoRA adapters additionally carry invocation fields; the parser must not trip on them.
        String json = "[{\"id\":1,\"path\":\"alora.gguf\",\"scale\":1.0,\"task_name\":\"\","
                + "\"prompt_prefix\":\"\",\"alora_invocation_string\":\"<invoke>\","
                + "\"alora_invocation_tokens\":[7,8]}]";
        List<LoraAdapter> result = parser.parse(json);
        assertThat(result, hasSize(1));
        assertThat(result.get(0).getPath(), is("alora.gguf"));
    }

    @Test
    public void testParse_multipleEntriesPreserveOrder() {
        String json =
                "[{\"id\":0,\"path\":\"a.gguf\",\"scale\":1.0}," + "{\"id\":1,\"path\":\"b.gguf\",\"scale\":0.0}]";
        List<LoraAdapter> result = parser.parse(json);
        assertThat(result, hasSize(2));
        assertThat(result.get(0).getPath(), is("a.gguf"));
        assertThat(result.get(1).getPath(), is("b.gguf"));
    }

    @Test
    public void testParse_emptyArray() {
        assertThat(parser.parse("[]"), is(empty()));
    }

    @Test
    public void testParse_nonArray() {
        assertThat(parser.parse("{\"success\":true}"), is(empty()));
    }

    @Test
    public void testParse_malformedJson() {
        assertThat(parser.parse("not json"), is(empty()));
    }

    // ------------------------------------------------------------------
    // toRequestJson(Map)
    // ------------------------------------------------------------------

    @Test
    public void testToRequestJson_singleEntry() {
        String json = parser.toRequestJson(Collections.singletonMap(0, 0.5f));
        assertThat(json, is("[{\"id\":0,\"scale\":0.5}]"));
    }

    @Test
    public void testToRequestJson_multipleEntriesInIterationOrder() {
        Map<Integer, Float> scales = new LinkedHashMap<>();
        scales.put(2, 0.25f);
        scales.put(0, 1.0f);
        String json = parser.toRequestJson(scales);
        assertThat(json, is("[{\"id\":2,\"scale\":0.25},{\"id\":0,\"scale\":1.0}]"));
    }

    @Test
    public void testToRequestJson_emptyMap() {
        assertThat(parser.toRequestJson(Collections.<Integer, Float>emptyMap()), is("[]"));
    }

    @Test
    public void testToRequestJson_nanScaleRejected() {
        assertThrows(
                IllegalArgumentException.class, () -> parser.toRequestJson(Collections.singletonMap(0, Float.NaN)));
    }

    @Test
    public void testToRequestJson_infiniteScaleRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> parser.toRequestJson(Collections.singletonMap(0, Float.POSITIVE_INFINITY)));
    }

    // ------------------------------------------------------------------
    // Round trip: request built here parses back on the C++ side contract
    // ------------------------------------------------------------------

    @Test
    public void testRequestJson_isValidAdapterArrayShape() {
        // The request shape is a subset of the response shape, so the parser reads it back.
        String json = parser.toRequestJson(Collections.singletonMap(3, 0.75f));
        List<LoraAdapter> parsed = parser.parse(json);
        assertThat(parsed, hasSize(1));
        assertThat(parsed.get(0).getId(), is(3));
        assertEquals(0.75f, parsed.get(0).getScale(), 0.0001f);
    }
}
