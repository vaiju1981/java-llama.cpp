// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

import java.io.File;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Integration tests for the runtime LoRA adapter control API
 * ({@link LlamaModel#getLoraAdapters()} / {@link LlamaModel#setLoraAdapters(Map)}), the typed
 * counterpart of the upstream {@code GET/POST /lora-adapters} endpoints.
 *
 * <p>CI ships no LoRA adapter GGUF, so these tests pin the adapter-less contract end to end:
 * the list round-trips as empty, and scale updates are accepted (upstream ignores ids that do
 * not correspond to a loaded adapter and rebuilds the — empty — adapter list). The
 * adapter-carrying paths of the wire format are covered model-free by
 * {@code LoraAdapterResponseParserTest} and the native C++ tests
 * ({@code ServerTaskResultGetLora}/{@code ParseLoraRequest}).
 */
@ClaudeGenerated(
        purpose = "Exercise the new getLoraAdapters/setLoraAdapters JNI round-trip against a real "
                + "model (adapter-less contract: empty list, accepted scale updates, stable "
                + "native state across repeated calls).")
public class RuntimeLoraIntegrationTest {

    private static LlamaModel model;

    @BeforeAll
    public static void setup() {
        Assumptions.assumeTrue(
                new File(TestConstants.REASONING_MODEL_PATH).exists(),
                "Reasoning model not found, skipping RuntimeLoraIntegrationTest");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.REASONING_MODEL_PATH)
                .setCtxSize(512)
                .setGpuLayers(gpuLayers)
                .setFit(false));
    }

    @AfterAll
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    @Test
    public void getLoraAdapters_withoutAdapters_returnsEmptyList() {
        assertThat(model.getLoraAdapters(), is(empty()));
    }

    @Test
    public void getLoraAdaptersJson_withoutAdapters_isEmptyJsonArray() {
        assertThat(model.getLoraAdaptersJson(), is("[]"));
    }

    @Test
    public void setLoraAdapters_unknownIdIsIgnored_listStaysEmpty() {
        // Upstream construct_lora_list only looks up ids of loaded adapters, so an unknown id
        // must be accepted silently rather than erroring.
        Map<Integer, Float> scales = new HashMap<>();
        scales.put(0, 0.5f);
        assertDoesNotThrow(() -> model.setLoraAdapters(scales));
        assertThat(model.getLoraAdapters(), is(empty()));
    }

    @Test
    public void setLoraAdapters_emptyMap_isAccepted() {
        assertDoesNotThrow(() -> model.setLoraAdapters(Collections.<Integer, Float>emptyMap()));
        assertThat(model.getLoraAdapters(), is(empty()));
    }

    @Test
    public void setLoraAdapter_singleConvenienceForm_isAccepted() {
        assertDoesNotThrow(() -> model.setLoraAdapter(0, 0.0f));
        assertThat(model.getLoraAdapters(), is(empty()));
    }
}
