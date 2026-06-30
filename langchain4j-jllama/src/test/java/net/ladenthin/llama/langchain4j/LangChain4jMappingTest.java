// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.output.FinishReason;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.value.ChatMessage;
import org.junit.jupiter.api.Test;

/** Model-free tests for the pure langchain4j&lt;-&gt;java-llama.cpp transforms. */
class LangChain4jMappingTest {

    @Test
    void mapsEveryRoleAndContent() {
        ChatRequest request =
                ChatRequest.builder()
                        .messages(
                                SystemMessage.from("you are terse"),
                                UserMessage.from("hi"),
                                AiMessage.from("hello"),
                                ToolExecutionResultMessage.from("call_1", "search", "42"))
                        .build();

        List<ChatMessage> messages = LangChain4jMapping.toJllamaRequest(request).getMessages();

        List<String> roles = new ArrayList<>();
        List<String> contents = new ArrayList<>();
        for (ChatMessage message : messages) {
            roles.add(message.getRole());
            contents.add(message.getContent());
        }
        assertThat(roles, contains("system", "user", "assistant", "tool"));
        assertThat(contents, contains("you are terse", "hi", "hello", "42"));
    }

    @Test
    void flattensMultimodalUserMessageToText() {
        ChatRequest request =
                ChatRequest.builder()
                        .messages(UserMessage.from(TextContent.from("Hello "), TextContent.from("world")))
                        .build();

        ChatMessage mapped = LangChain4jMapping.toJllamaRequest(request).getMessages().get(0);

        assertThat(mapped.getRole(), is("user"));
        assertThat(mapped.getContent(), is("Hello world"));
    }

    @Test
    void appliesSamplingParametersToInferenceJson() {
        ChatRequest request =
                ChatRequest.builder()
                        .messages(UserMessage.from("hi"))
                        .temperature(0.3)
                        .topK(40)
                        .maxOutputTokens(64)
                        .frequencyPenalty(0.5)
                        .presencePenalty(0.25)
                        .stopSequences(Arrays.asList("STOP"))
                        .build();

        String json = LangChain4jMapping.toStreamingParameters(request).toString();

        assertThat(json, containsString("\"temperature\""));
        assertThat(json, containsString("\"top_k\""));
        assertThat(json, containsString("\"n_predict\""));
        assertThat(json, containsString("\"frequency_penalty\""));
        assertThat(json, containsString("\"presence_penalty\""));
        assertThat(json, containsString("\"stop\""));
        // Messages must survive into the streaming parameter blob too.
        assertThat(json, containsString("hi"));
    }

    @Test
    void mapsFinishReasonStrings() {
        assertThat(LangChain4jMapping.toFinishReason("stop"), is(FinishReason.STOP));
        assertThat(LangChain4jMapping.toFinishReason("length"), is(FinishReason.LENGTH));
        assertThat(LangChain4jMapping.toFinishReason("tool_calls"), is(FinishReason.TOOL_EXECUTION));
        assertThat(LangChain4jMapping.toFinishReason("content_filter"), is(FinishReason.CONTENT_FILTER));
        assertThat(LangChain4jMapping.toFinishReason("something_new"), is(FinishReason.OTHER));
        // No choices / absent reason is the normal terminal state.
        assertThat(LangChain4jMapping.toFinishReason(null), is(FinishReason.STOP));
    }

    @Test
    void rerankScoresAlignToInputOrderNotResponseOrder() {
        // Native results arrive out of order; "index" is the input position.
        String json =
                "[{\"document\":\"b\",\"index\":1,\"score\":0.9},"
                        + "{\"document\":\"a\",\"index\":0,\"score\":0.1}]";

        double[] scores = LangChain4jMapping.parseRerankScores(json, 2);

        assertThat(scores.length, is(2));
        assertThat(scores[0], is(0.1));
        assertThat(scores[1], is(0.9));
    }

    @Test
    void rerankScoresDefaultToZeroForMissingEntries() {
        double[] scores = LangChain4jMapping.parseRerankScores("[]", 3);

        assertThat(scores.length, is(3));
        assertThat(scores[0], is(0.0));
        assertThat(scores[1], is(0.0));
        assertThat(scores[2], is(0.0));
    }

    @Test
    void rerankScoresFallBackToArrayOrderWhenIndexAbsent() {
        // No "index" field: array position is used, so scores are not silently all-zero.
        String json = "[{\"document\":\"a\",\"score\":0.7},{\"document\":\"b\",\"score\":0.2}]";

        double[] scores = LangChain4jMapping.parseRerankScores(json, 2);

        assertThat(scores[0], is(0.7));
        assertThat(scores[1], is(0.2));
    }
}
