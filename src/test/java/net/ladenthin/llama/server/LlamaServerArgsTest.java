// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify LlamaServerArgs parses long/short flags, applies defaults, derives the model alias from the "
                + "model path, and rejects unknown flags, missing values, malformed integers and a missing --model.")
public class LlamaServerArgsTest {

    @Test
    public void minimalArgsApplyDefaults() {
        LlamaServerConfig config = LlamaServerArgs.parse(new String[] {"--model", "models/Qwen3-0.6B.gguf"});
        assertThat(config.getModelPath(), is("models/Qwen3-0.6B.gguf"));
        assertThat(config.getHost(), is(LlamaServerArgs.DEFAULT_HOST));
        assertThat(config.getPort(), is(LlamaServerArgs.DEFAULT_PORT));
        assertThat(config.getCtxSize(), is(0));
        assertThat(config.getGpuLayers(), is(0));
        assertThat(config.getThreads(), is(0));
        assertThat(config.isEmbedding(), is(false));
        // Alias defaults to the model file name.
        assertThat(config.getModelAlias(), is("Qwen3-0.6B.gguf"));
    }

    @Test
    public void allLongFlagsParsed() {
        LlamaServerConfig config = LlamaServerArgs.parse(new String[] {
            "--model", "m.gguf",
            "--host", "0.0.0.0",
            "--port", "9090",
            "--ctx-size", "4096",
            "--n-gpu-layers", "99",
            "--threads", "8",
            "--model-alias", "my-model",
            "--embedding"
        });
        assertThat(config.getModelPath(), is("m.gguf"));
        assertThat(config.getHost(), is("0.0.0.0"));
        assertThat(config.getPort(), is(9090));
        assertThat(config.getCtxSize(), is(4096));
        assertThat(config.getGpuLayers(), is(99));
        assertThat(config.getThreads(), is(8));
        assertThat(config.getModelAlias(), is("my-model"));
        assertThat(config.isEmbedding(), is(true));
    }

    @Test
    public void shortFlagsParsed() {
        LlamaServerConfig config = LlamaServerArgs.parse(
                new String[] {"-m", "m.gguf", "-p", "1234", "-c", "512", "-ngl", "10", "-t", "4"});
        assertThat(config.getPort(), is(1234));
        assertThat(config.getCtxSize(), is(512));
        assertThat(config.getGpuLayers(), is(10));
        assertThat(config.getThreads(), is(4));
    }

    @Test
    public void aliasDerivedFromNestedPath() {
        LlamaServerConfig config = LlamaServerArgs.parse(new String[] {"-m", "/opt/models/Llama-3.gguf"});
        assertThat(config.getModelAlias(), is("Llama-3.gguf"));
    }

    @Test
    public void missingModelThrows() {
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> LlamaServerArgs.parse(new String[] {}));
        assertThat(ex.getMessage(), containsString("--model"));
    }

    @Test
    public void unknownFlagThrows() {
        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class, () -> LlamaServerArgs.parse(new String[] {"-m", "m.gguf", "--bogus"}));
        assertThat(ex.getMessage(), containsString("Unknown argument: --bogus"));
    }

    @Test
    public void missingValueThrows() {
        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class, () -> LlamaServerArgs.parse(new String[] {"-m", "m.gguf", "--port"}));
        assertThat(ex.getMessage(), containsString("Missing value for --port"));
    }

    @Test
    public void nonIntegerPortThrows() {
        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class,
                () -> LlamaServerArgs.parse(new String[] {"-m", "m.gguf", "--port", "abc"}));
        assertThat(ex.getMessage(), containsString("expects an integer"));
    }

    @Test
    public void helpRequestedDetection() {
        assertThat(LlamaServerArgs.isHelpRequested(new String[] {"-h"}), is(true));
        assertThat(LlamaServerArgs.isHelpRequested(new String[] {"--help"}), is(true));
        assertThat(LlamaServerArgs.isHelpRequested(new String[] {"--model", "m.gguf"}), is(false));
    }

    @Test
    public void usageMentionsEndpointsAndRequiredFlag() {
        String usage = LlamaServerArgs.usage();
        assertThat(usage, containsString("--model"));
        assertThat(usage, containsString("/v1/chat/completions"));
        assertThat(usage, containsString("/v1/embeddings"));
    }
}
