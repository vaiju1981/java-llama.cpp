// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.ObjectMapper;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link ModelMeta} typed getters.
 * Constructs {@code ModelMeta} directly from JSON strings — no native library or model file required.
 */
@ClaudeGenerated(
        purpose = "Verify that ModelMeta typed getters map correctly from the underlying JsonNode, "
                + "including the new architecture and name fields from GGUF general.* metadata.")
public class ModelMetaTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private ModelMeta parse(String json) throws Exception {
        return new ModelMeta(MAPPER.readTree(json));
    }

    @Test
    public void testNumericGetters() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":32016,\"n_ctx_train\":16384,"
                + "\"n_embd\":4096,\"n_params\":6738546688,\"size\":2825274880,"
                + "\"modalities\":{\"vision\":false,\"audio\":false},"
                + "\"architecture\":\"llama\",\"name\":\"CodeLlama-7B\"}");

        assertThat(meta.getVocabType(), is(1));
        assertThat(meta.getNVocab(), is(32016));
        assertThat(meta.getNCtxTrain(), is(16384));
        assertThat(meta.getNEmbd(), is(4096));
        assertThat(meta.getNParams(), is(6738546688L));
        assertThat(meta.getSize(), is(2825274880L));
    }

    @Test
    public void testModalityGetters() throws Exception {
        ModelMeta textOnly = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096,"
                + "\"n_embd\":512,\"n_params\":1000000,\"size\":500000,"
                + "\"modalities\":{\"vision\":false,\"audio\":false},"
                + "\"architecture\":\"llama\",\"name\":\"\"}");
        assertThat(textOnly.supportsVision(), is(false));
        assertThat(textOnly.supportsAudio(), is(false));

        ModelMeta multimodal = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096,"
                + "\"n_embd\":512,\"n_params\":1000000,\"size\":500000,"
                + "\"modalities\":{\"vision\":true,\"audio\":true},"
                + "\"architecture\":\"gemma3\",\"name\":\"Gemma-3\"}");
        assertThat(multimodal.supportsVision(), is(true));
        assertThat(multimodal.supportsAudio(), is(true));
    }

    @Test
    public void testGetArchitecture() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":32016,\"n_ctx_train\":16384,"
                + "\"n_embd\":4096,\"n_params\":6738546688,\"size\":2825274880,"
                + "\"modalities\":{\"vision\":false,\"audio\":false},"
                + "\"architecture\":\"llama\",\"name\":\"CodeLlama-7B\"}");

        assertThat(meta.getArchitecture(), is("llama"));
    }

    @Test
    public void testGetModelName() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":32016,\"n_ctx_train\":16384,"
                + "\"n_embd\":4096,\"n_params\":6738546688,\"size\":2825274880,"
                + "\"modalities\":{\"vision\":false,\"audio\":false},"
                + "\"architecture\":\"mistral\",\"name\":\"Mistral-7B-v0.1\"}");

        assertThat(meta.getModelName(), is("Mistral-7B-v0.1"));
    }

    @Test
    public void testGetArchitectureEmptyWhenAbsent() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096,"
                + "\"n_embd\":512,\"n_params\":1000000,\"size\":500000,"
                + "\"modalities\":{\"vision\":false,\"audio\":false}}");

        assertThat(meta.getArchitecture(), is(""));
    }

    @Test
    public void testGetModelNameEmptyWhenAbsent() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096,"
                + "\"n_embd\":512,\"n_params\":1000000,\"size\":500000,"
                + "\"modalities\":{\"vision\":false,\"audio\":false}}");

        assertThat(meta.getModelName(), is(""));
    }

    @Test
    public void testGetArchitectureVariousModels() throws Exception {
        for (String arch : new String[] {"llama", "gemma3", "mistral", "falcon", "phi3"}) {
            ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096,"
                    + "\"n_embd\":512,\"n_params\":1000000,\"size\":500000,"
                    + "\"modalities\":{\"vision\":false,\"audio\":false},"
                    + "\"architecture\":\""
                    + arch + "\",\"name\":\"\"}");
            assertThat(meta.getArchitecture(), is(arch));
        }
    }

    @Test
    public void testToStringContainsNewFields() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":32016,\"n_ctx_train\":16384,"
                + "\"n_embd\":4096,\"n_params\":6738546688,\"size\":2825274880,"
                + "\"modalities\":{\"vision\":false,\"audio\":false},"
                + "\"architecture\":\"llama\",\"name\":\"CodeLlama-7B\"}");

        String json = meta.toString();
        assertThat(json, containsString("\"architecture\""));
        assertThat(json, containsString("\"name\""));
        assertThat(json, containsString("\"llama\""));
        assertThat(json, containsString("\"CodeLlama-7B\""));
    }
}
