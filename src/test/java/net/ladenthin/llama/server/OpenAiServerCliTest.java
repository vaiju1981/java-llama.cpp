// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link OpenAiServerCli}: parsing of long/short/alias flags, defaults, model-id
 * derivation, rejection of bad input, and the {@link OpenAiServerCli.Options#toServerConfig()}
 * projection (auth + context-derived token budgets). Pure — no socket and no native model.
 */
public class OpenAiServerCliTest {

    @Test
    public void minimalArgsApplyDefaults() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse("--model", "models/Qwen3-0.6B.gguf");
        assertThat(options.getModelPath(), is("models/Qwen3-0.6B.gguf"));
        assertThat(options.getHost(), is(OpenAiServerCli.DEFAULT_HOST));
        assertThat(options.getPort(), is(OpenAiServerCli.DEFAULT_PORT));
        assertThat(options.getCtxSize(), is(0));
        assertThat(options.getGpuLayers(), is(0));
        assertThat(options.getThreads(), is(0));
        assertThat(options.getParallel(), is(0));
        assertThat(options.isEmbedding(), is(false));
        assertThat(options.getApiKey(), is((String) null));
        // Model id defaults to the model file name.
        assertThat(options.getModelId(), is("Qwen3-0.6B.gguf"));
    }

    @Test
    public void allLongFlagsParsed() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse(
                "--model",
                "m.gguf",
                "--host",
                "0.0.0.0",
                "--port",
                "9090",
                "--ctx-size",
                "4096",
                "--n-gpu-layers",
                "99",
                "--threads",
                "8",
                "--parallel",
                "2",
                "--model-id",
                "my-model",
                "--api-key",
                "secret",
                "--embedding");
        assertThat(options.getModelPath(), is("m.gguf"));
        assertThat(options.getHost(), is("0.0.0.0"));
        assertThat(options.getPort(), is(9090));
        assertThat(options.getCtxSize(), is(4096));
        assertThat(options.getGpuLayers(), is(99));
        assertThat(options.getThreads(), is(8));
        assertThat(options.getParallel(), is(2));
        assertThat(options.getModelId(), is("my-model"));
        assertThat(options.getApiKey(), is("secret"));
        assertThat(options.isEmbedding(), is(true));
    }

    @Test
    public void shortFlagsParsed() {
        OpenAiServerCli.Options options =
                OpenAiServerCli.parse("-m", "m.gguf", "-p", "1234", "-c", "512", "-ngl", "10", "-t", "4");
        assertThat(options.getPort(), is(1234));
        assertThat(options.getCtxSize(), is(512));
        assertThat(options.getGpuLayers(), is(10));
        assertThat(options.getThreads(), is(4));
    }

    @Test
    public void legacyAliasFlagsAccepted() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse(
                "-m", "m.gguf", "--ctx", "256", "--gpu-layers", "5", "--model-alias", "aliased", "--embeddings");
        assertThat(options.getCtxSize(), is(256));
        assertThat(options.getGpuLayers(), is(5));
        assertThat(options.getModelId(), is("aliased"));
        assertThat(options.isEmbedding(), is(true));
    }

    @Test
    public void modelIdDerivedFromNestedPath() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse("-m", "/opt/models/Llama-3.gguf");
        assertThat(options.getModelId(), is("Llama-3.gguf"));
    }

    @Test
    public void explicitModelIdOverridesDerivation() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse("-m", "/opt/models/Llama-3.gguf", "--model-id", "x");
        assertThat(options.getModelId(), is("x"));
    }

    @Test
    public void missingModelThrows() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, OpenAiServerCli::parse);
        assertThat(ex.getMessage(), containsString("--model"));
    }

    @Test
    public void unknownFlagThrows() {
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> OpenAiServerCli.parse("-m", "m.gguf", "--bogus"));
        assertThat(ex.getMessage(), containsString("Unknown argument: --bogus"));
    }

    @Test
    public void missingValueThrows() {
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> OpenAiServerCli.parse("-m", "m.gguf", "--port"));
        assertThat(ex.getMessage(), containsString("Missing value for --port"));
    }

    @Test
    public void nonIntegerPortThrows() {
        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class, () -> OpenAiServerCli.parse("-m", "m.gguf", "--port", "abc"));
        assertThat(ex.getMessage(), containsString("expects an integer"));
    }

    @Test
    public void helpRequestedDetection() {
        assertThat(OpenAiServerCli.isHelpRequested("-h"), is(true));
        assertThat(OpenAiServerCli.isHelpRequested("--help"), is(true));
        assertThat(OpenAiServerCli.isHelpRequested("--model", "m.gguf"), is(false));
    }

    @Test
    public void usageMentionsEndpointsAndRequiredFlag() {
        String usage = OpenAiServerCli.usage();
        assertThat(usage, containsString("--model"));
        assertThat(usage, containsString("/v1/chat/completions"));
        assertThat(usage, containsString("/v1/embeddings"));
        assertThat(usage, containsString("/health"));
    }

    @Test
    public void serverConfigCarriesHostPortAndModelId() {
        OpenAiServerConfig config = OpenAiServerCli.parse("-m", "m.gguf", "--host", "0.0.0.0", "-p", "1234")
                .toServerConfig();
        assertThat(config.getHost(), is("0.0.0.0"));
        assertThat(config.getPort(), is(1234));
        assertThat(config.getModelId(), is("m.gguf"));
        assertThat(config.isAuthenticationEnabled(), is(false));
    }

    @Test
    public void apiKeyEnablesAuthInServerConfig() {
        OpenAiServerConfig config =
                OpenAiServerCli.parse("-m", "m.gguf", "--api-key", "secret").toServerConfig();
        assertThat(config.isAuthenticationEnabled(), is(true));
        assertThat(config.getApiKey(), is("secret"));
    }

    @Test
    public void contextSizeDerivesAdvertisedTokenBudgets() {
        // ctx 8192 -> output capped at the 2048 default, input = 8192 - 2048 = 6144 (README example).
        OpenAiServerConfig config =
                OpenAiServerCli.parse("-m", "m.gguf", "-c", "8192").toServerConfig();
        assertThat(config.getMaxOutputTokens(), is(2048));
        assertThat(config.getMaxInputTokens(), is(6144));
    }

    @Test
    public void modelParametersIncludeModelPath() {
        String json =
                OpenAiServerCli.parse("-m", "models/m.gguf").toModelParameters().toString();
        assertThat(json, containsString("models/m.gguf"));
    }
}
