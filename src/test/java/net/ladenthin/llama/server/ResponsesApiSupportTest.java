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
 * Unit tests for {@link ResponsesApiSupport}: the OpenAI Responses &harr; OpenAI chat request/response
 * translation (input items, instructions, function tools, function_call output items). Pure — no model.
 */
public class ResponsesApiSupportTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static JsonNode read(String json) throws IOException {
        return MAPPER.readTree(json);
    }

    @Test
    public void isStreamingDefaultsFalse() throws IOException {
        assertThat(ResponsesApiSupport.isStreaming(read("{}")), is(false));
        assertThat(ResponsesApiSupport.isStreaming(read("{\"stream\":true}")), is(true));
    }

    @Test
    public void requestMapsInstructionsStringInputAndTools() throws IOException {
        JsonNode openAi = ResponsesApiSupport.toOpenAiChatRequest(read("{\"model\":\"m\","
                + "\"instructions\":\"be brief\",\"input\":\"hi\",\"max_output_tokens\":32,"
                + "\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}]}"));
        assertThat(openAi.path("messages").get(0).path("role").asText(), is("system"));
        assertThat(openAi.path("messages").get(0).path("content").asText(), is("be brief"));
        assertThat(openAi.path("messages").get(1).path("role").asText(), is("user"));
        assertThat(openAi.path("messages").get(1).path("content").asText(), is("hi"));
        // max_output_tokens -> max_tokens.
        assertThat(openAi.path("max_tokens").asInt(), is(32));
        // Responses function tool (flat) -> OpenAI nested function tool.
        assertThat(openAi.path("tools").get(0).path("function").path("name").asText(), is("f"));
        assertThat(
                openAi.path("tools")
                        .get(0)
                        .path("function")
                        .path("parameters")
                        .path("type")
                        .asText(),
                is("object"));
    }

    @Test
    public void requestMapsInputArrayMessageAndFunctionCallItems() throws IOException {
        String responses = "{\"input\":["
                + "{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"q\"}]},"
                + "{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"f\",\"arguments\":\"{}\"},"
                + "{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":\"ok\"}]}";
        JsonNode openAi = ResponsesApiSupport.toOpenAiChatRequest(read(responses));
        assertThat(openAi.path("messages").get(0).path("content").asText(), is("q"));
        // function_call -> assistant tool_calls
        JsonNode toolCall = openAi.path("messages").get(1).path("tool_calls").get(0);
        assertThat(toolCall.path("id").asText(), is("c1"));
        assertThat(toolCall.path("function").path("name").asText(), is("f"));
        // function_call_output -> role:"tool" message
        assertThat(openAi.path("messages").get(2).path("role").asText(), is("tool"));
        assertThat(openAi.path("messages").get(2).path("tool_call_id").asText(), is("c1"));
        assertThat(openAi.path("messages").get(2).path("content").asText(), is("ok"));
    }

    @Test
    public void responseWrapsOutputMessageWithOutputTextAndUsage() throws IOException {
        String openAi = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"hello\"},"
                + "\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1}}";
        JsonNode out = read(ResponsesApiSupport.toResponsesResponse(openAi, "m", "resp_1"));
        assertThat(out.path("object").asText(), is("response"));
        assertThat(out.path("status").asText(), is("completed"));
        JsonNode messageItem = out.path("output").get(0);
        assertThat(messageItem.path("type").asText(), is("message"));
        assertThat(messageItem.path("content").get(0).path("type").asText(), is("output_text"));
        assertThat(messageItem.path("content").get(0).path("text").asText(), is("hello"));
        assertThat(out.path("usage").path("input_tokens").asInt(), is(4));
        assertThat(out.path("usage").path("total_tokens").asInt(), is(5));
    }

    @Test
    public void responseEmitsFunctionCallItemsForToolCalls() throws IOException {
        String openAi = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"\","
                + "\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\","
                + "\"arguments\":\"{\\\"a\\\":1}\"}}]},\"finish_reason\":\"tool_calls\"}]}";
        JsonNode out = read(ResponsesApiSupport.toResponsesResponse(openAi, "m", "resp_1"));
        // output[0] is the (empty) message, output[1] is the function_call item.
        JsonNode functionCall = out.path("output").get(1);
        assertThat(functionCall.path("type").asText(), is("function_call"));
        assertThat(functionCall.path("call_id").asText(), is("c1"));
        assertThat(functionCall.path("name").asText(), is("f"));
        assertThat(functionCall.path("arguments").asText(), is("{\"a\":1}"));
    }
}
