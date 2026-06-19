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

/**
 * Unit tests for {@link OllamaApiSupport}: discovery bodies and the Ollama&harr;OpenAI chat request,
 * response and NDJSON streaming translation. Pure — no model.
 */
public class OllamaApiSupportTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static JsonNode read(String json) throws IOException {
        return MAPPER.readTree(json);
    }

    @Test
    public void versionJsonCarriesAVersion() throws IOException {
        assertThat(read(OllamaApiSupport.versionJson()).path("version").asText(), is(OllamaApiSupport.OLLAMA_VERSION));
    }

    @Test
    public void tagsJsonListsTheModel() throws IOException {
        JsonNode out = read(OllamaApiSupport.tagsJson("local-qwen"));
        assertThat(out.path("models").get(0).path("name").asText(), is("local-qwen"));
        assertThat(out.path("models").get(0).path("model").asText(), is("local-qwen"));
    }

    @Test
    public void showJsonAdvertisesCapabilitiesAndContextLength() throws IOException {
        JsonNode out = read(OllamaApiSupport.showJson("local-qwen", 8192, false));
        assertThat(out.path("model_info").path("llama.context_length").asInt(), is(8192));
        String capabilities = out.path("capabilities").toString();
        assertThat(capabilities.contains("completion"), is(true));
        assertThat(capabilities.contains("tools"), is(true));
        assertThat(capabilities.contains("vision"), is(false));
    }

    @Test
    public void showJsonAddsVisionCapabilityWhenEnabled() throws IOException {
        JsonNode out = read(OllamaApiSupport.showJson("m", 4096, true));
        assertThat(out.path("capabilities").toString().contains("vision"), is(true));
    }

    @Test
    public void isStreamingDefaultsTrueAndHonoursExplicitFalse() throws IOException {
        assertThat(OllamaApiSupport.isStreaming(read("{}")), is(true));
        assertThat(OllamaApiSupport.isStreaming(read("{\"stream\":false}")), is(false));
        assertThat(OllamaApiSupport.isStreaming(read("{\"stream\":true}")), is(true));
    }

    @Test
    public void toOpenAiChatRequestMapsMessagesToolsAndOptions() throws IOException {
        JsonNode openAi = OllamaApiSupport.toOpenAiChatRequest(read("{\"model\":\"m\","
                + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                + "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\"}}],"
                + "\"options\":{\"temperature\":0.4,\"num_predict\":64}}"));
        assertThat(openAi.path("messages").get(0).path("content").asText(), is("hi"));
        assertThat(openAi.path("tools").get(0).path("function").path("name").asText(), is("f"));
        assertThat(openAi.path("temperature").asDouble(), is(0.4));
        // num_predict maps to OpenAI max_tokens.
        assertThat(openAi.path("max_tokens").asInt(), is(64));
    }

    @Test
    public void toOpenAiChatRequestStringifiesAssistantToolCallArguments() throws IOException {
        JsonNode openAi = OllamaApiSupport.toOpenAiChatRequest(
                read(
                        "{\"messages\":["
                                + "{\"role\":\"assistant\",\"tool_calls\":[{\"function\":{\"name\":\"f\",\"arguments\":{\"a\":1}}}]}]}"));
        JsonNode arguments = openAi.path("messages")
                .get(0)
                .path("tool_calls")
                .get(0)
                .path("function")
                .path("arguments");
        // Ollama sends an object; OpenAI requires a JSON-encoded string.
        assertThat(arguments.isTextual(), is(true));
        assertThat(read(arguments.asText()).path("a").asInt(), is(1));
    }

    @Test
    public void toOpenAiChatRequestMapsFormatJsonToResponseFormat() throws IOException {
        JsonNode openAi = OllamaApiSupport.toOpenAiChatRequest(
                read("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"format\":\"json\"}"));
        assertThat(openAi.path("response_format").path("type").asText(), is("json_object"));
    }

    @Test
    public void toOllamaChatResponseExtractsContentAndCountsAndDone() throws IOException {
        String openAi = "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"hello\"},"
                + "\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3}}";
        JsonNode out = read(OllamaApiSupport.toOllamaChatResponse(openAi, "m"));
        assertThat(out.path("message").path("content").asText(), is("hello"));
        assertThat(out.path("done").asBoolean(), is(true));
        assertThat(out.path("done_reason").asText(), is("stop"));
        assertThat(out.path("prompt_eval_count").asInt(), is(7));
        assertThat(out.path("eval_count").asInt(), is(3));
    }

    @Test
    public void toOllamaChatResponseConvertsToolCallArgumentsToObject() throws IOException {
        String openAi = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":null,"
                + "\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\","
                + "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}]}}]}";
        JsonNode out = read(OllamaApiSupport.toOllamaChatResponse(openAi, "m"));
        JsonNode arguments =
                out.path("message").path("tool_calls").get(0).path("function").path("arguments");
        // OpenAI string arguments become an Ollama object.
        assertThat(arguments.isObject(), is(true));
        assertThat(arguments.path("city").asText(), is("Paris"));
    }

    @Test
    public void toOllamaContentLineEmitsDeltaAndSkipsEmpty() throws IOException {
        String line = OllamaApiSupport.toOllamaContentLine("{\"choices\":[{\"delta\":{\"content\":\"he\"}}]}", "m");
        assertThat(line.endsWith("\n"), is(true));
        JsonNode parsed = read(line.trim());
        assertThat(parsed.path("message").path("content").asText(), is("he"));
        assertThat(parsed.path("done").asBoolean(), is(false));
        // Role-only / empty-content chunks emit nothing.
        assertThat(
                OllamaApiSupport.toOllamaContentLine("{\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}", "m"),
                is((String) null));
    }

    @Test
    public void toOllamaDoneLineCarriesAccumulatedToolCalls() throws IOException {
        ToolCallDeltaAccumulator accumulator = new ToolCallDeltaAccumulator();
        accumulator.accept("{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\","
                + "\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":1}\"}}]}}]}");
        JsonNode out = read(OllamaApiSupport.toOllamaDoneLine("m", accumulator).trim());
        assertThat(out.path("done").asBoolean(), is(true));
        JsonNode arguments =
                out.path("message").path("tool_calls").get(0).path("function").path("arguments");
        assertThat(arguments.isObject(), is(true));
        assertThat(arguments.path("a").asInt(), is(1));
    }
}
