// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.json.ChatResponseParser;
import net.ladenthin.llama.parameters.ChatRequest;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify ChatResponseParser.parseResponse maps the OAI-compatible chat completion JSON "
                + "into ChatResponse / ChatChoice / ChatMessage / ToolCall, surfaces Usage and Timings, "
                + "and falls back gracefully on malformed input.")
public class ChatResponseTest {

    private final ChatResponseParser parser = new ChatResponseParser();

    @Test
    public void parsesPlainAssistantReply() {
        String json = "{\"id\":\"chatcmpl-1\","
                + "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"Hello!\"},"
                + "\"finish_reason\":\"stop\"}],"
                + "\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5,\"total_tokens\":17},"
                + "\"timings\":{\"prompt_n\":12,\"prompt_ms\":100.0,\"prompt_per_second\":120.0,"
                + "\"predicted_n\":5,\"predicted_ms\":50.0,\"predicted_per_second\":100.0}}";
        ChatResponse r = parser.parseResponse(json);

        assertThat(r.getId(), is("chatcmpl-1"));
        assertThat(r.getChoices(), hasSize(1));
        ChatChoice c = r.getChoices().get(0);
        assertThat(c.getIndex(), is(0));
        assertThat(c.getMessage().getRole(), is("assistant"));
        assertThat(c.getMessage().getContent(), is("Hello!"));
        assertThat(c.getFinishReason(), is("stop"));
        assertThat(c.getMessage().getToolCalls(), is(empty()));

        assertThat(r.getUsage().getPromptTokens(), is(12L));
        assertThat(r.getUsage().getCompletionTokens(), is(5L));
        assertThat(r.getUsage().getTotalTokens(), is(17L));

        assertThat(r.getTimings().getPromptN(), is(12L));
        assertEquals(100.0, r.getTimings().getPromptMs(), 1e-9);
        assertEquals(100.0, r.getTimings().getPredictedPerSecond(), 1e-9);

        assertThat(r.getFirstContent(), is("Hello!"));
    }

    @Test
    public void parsesToolCalls() {
        String json = "{\"id\":\"x\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                + "\"content\":\"\",\"tool_calls\":["
                + "{\"id\":\"call_a\",\"type\":\"function\",\"function\":"
                + "{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Berlin\\\"}\"}},"
                + "{\"id\":\"call_b\",\"type\":\"function\",\"function\":"
                + "{\"name\":\"get_time\",\"arguments\":\"{}\"}}"
                + "]},\"finish_reason\":\"tool_calls\"}],"
                + "\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":7}}";
        ChatResponse r = parser.parseResponse(json);
        ChatMessage m = r.getFirstMessage().orElseThrow();
        assertThat(m.getRole(), is("assistant"));
        List<ToolCall> tc = m.getToolCalls();
        assertThat(tc, hasSize(2));
        assertThat(tc.get(0).getId(), is("call_a"));
        assertThat(tc.get(0).getName(), is("get_weather"));
        assertThat(tc.get(0).getArgumentsJson(), is("{\"city\":\"Berlin\"}"));
        assertThat(tc.get(1).getName(), is("get_time"));
        assertThat(r.getChoices().get(0).getFinishReason(), is("tool_calls"));
    }

    @Test
    public void parsesObjectShapedArguments() {
        // Some upstream variants emit arguments as an object instead of a string
        String json = "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                + "\"tool_calls\":[{\"id\":\"x\",\"type\":\"function\",\"function\":"
                + "{\"name\":\"f\",\"arguments\":{\"a\":1,\"b\":2}}}]},"
                + "\"finish_reason\":\"tool_calls\"}]}";
        ChatResponse r = parser.parseResponse(json);
        String args = r.getFirstMessage().orElseThrow().getToolCalls().get(0).getArgumentsJson();
        // exact text isn't guaranteed, but must contain both fields
        assertThat("expected serialized object, got: " + args, args, containsString("\"a\":1"));
        assertThat(args, containsString("\"b\":2"));
    }

    @Test
    public void malformedInputYieldsEmptyResponse() {
        ChatResponse r = parser.parseResponse("{not json");
        assertThat(r.getId(), is(""));
        assertThat(r.getChoices(), is(empty()));
        assertThat(r.getUsage().getTotalTokens(), is(0L));
    }

    @Test
    public void rawJsonIsPreserved() {
        String json = "{\"id\":\"chatcmpl-raw\",\"choices\":[]}";
        ChatResponse r = parser.parseResponse(json);
        // Assert on content (not just non-null) so the empty-string return mutant is killed.
        assertThat(r.getRawJson(), containsString("chatcmpl-raw"));
    }

    @Test
    public void buildMessagesJsonRoundTripsToolTurns() {
        ChatRequest req = ChatRequest.empty()
                .appendMessage("system", "be terse")
                .appendMessage("user", "two plus two?")
                .appendMessage(ChatMessage.assistantToolCalls(
                        "", java.util.Collections.singletonList(new ToolCall("c1", "add", "{\"a\":2,\"b\":2}"))))
                .appendMessage(ChatMessage.toolResult("c1", "4"));

        String msgs = req.buildMessagesJson();
        assertThat(msgs, msgs, containsString("\"tool_calls\""));
        assertThat(msgs, msgs, containsString("\"tool_call_id\":\"c1\""));
        assertThat(msgs, msgs, containsString("\"name\":\"add\""));
    }

    @Test
    public void buildToolsJsonEmptyWhenNoTools() {
        ChatRequest req = ChatRequest.empty().appendMessage("user", "hi");
        assertThat(req.buildToolsJson().isPresent(), is(false));
    }

    @Test
    public void buildToolsJsonInlinesParameterSchema() {
        ChatRequest req = ChatRequest.empty()
                .appendTool(new ToolDefinition(
                        "echo", "Echo a string", "{\"type\":\"object\",\"properties\":{\"s\":{\"type\":\"string\"}}}"));
        String tools = req.buildToolsJson().orElseThrow();
        assertThat(tools, tools, containsString("\"type\":\"function\""));
        assertThat(tools, tools, containsString("\"name\":\"echo\""));
        assertThat(tools, tools, containsString("\"properties\""));
    }
}
