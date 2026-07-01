// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import com.fasterxml.jackson.databind.JsonNode;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ContentType;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import java.io.IOException;
import java.util.List;
import net.ladenthin.llama.json.RerankResponseParser;
import net.ladenthin.llama.parameters.InferenceParameters;

/**
 * Pure (model-free) translation between langchain4j chat types and java-llama.cpp parameters.
 *
 * <p>Every method here is a deterministic data transform with no JNI and no loaded model, so the
 * mapping is unit-testable on its own (see {@code LangChain4jMappingTest}). The adapters keep the
 * live-model calls; this class only reshapes their inputs and outputs.
 */
final class LangChain4jMapping {

    private LangChain4jMapping() {}

    /**
     * Build a java-llama.cpp typed chat request from a langchain4j chat request. Messages map by
     * role; sampling parameters ({@code temperature}/{@code topP}/{@code topK}/{@code
     * maxOutputTokens}/{@code stopSequences}) ride along as an inference customizer.
     */
    static net.ladenthin.llama.parameters.ChatRequest toJllamaRequest(ChatRequest request) {
        net.ladenthin.llama.parameters.ChatRequest jllama =
                net.ladenthin.llama.parameters.ChatRequest.empty();
        for (ChatMessage message : request.messages()) {
            jllama = jllama.appendMessage(toJllamaMessage(message));
        }
        return jllama.withInferenceCustomizer(params -> applySampling(params, request));
    }

    /**
     * Build the streaming inference parameters (messages JSON + sampling) for {@code generateChat}.
     * Shares {@link #toJllamaRequest(ChatRequest)} so blocking and streaming stay in lockstep.
     */
    static InferenceParameters toStreamingParameters(ChatRequest request) {
        net.ladenthin.llama.parameters.ChatRequest jllama = toJllamaRequest(request);
        InferenceParameters params =
                InferenceParameters.empty().withMessagesJson(jllama.buildMessagesJson());
        return jllama.applyCustomizer(params);
    }

    /** Wrap a java-llama.cpp chat result as a langchain4j {@link ChatResponse}. */
    static ChatResponse toLangChainResponse(net.ladenthin.llama.value.ChatResponse response) {
        ChatResponse.Builder builder =
                ChatResponse.builder().aiMessage(AiMessage.from(response.getFirstContent()));
        net.ladenthin.llama.value.Usage usage = response.getUsage();
        if (usage != null) {
            builder.tokenUsage(
                    new TokenUsage((int) usage.getPromptTokens(), (int) usage.getCompletionTokens()));
        }
        List<net.ladenthin.llama.value.ChatChoice> choices = response.getChoices();
        String finishReason = choices.isEmpty() ? null : choices.get(0).getFinishReason();
        return builder.finishReason(toFinishReason(finishReason)).build();
    }

    /**
     * Map java-llama.cpp's OpenAI-style finish-reason string to the langchain4j enum. A {@code null}
     * (no choices / reason absent) is treated as a normal {@code STOP}; an unrecognized value maps to
     * {@code OTHER} rather than guessing.
     */
    static FinishReason toFinishReason(String reason) {
        if (reason == null) {
            return FinishReason.STOP;
        }
        switch (reason) {
            case "stop":
                return FinishReason.STOP;
            case "length":
                return FinishReason.LENGTH;
            case "tool_calls":
                return FinishReason.TOOL_EXECUTION;
            case "content_filter":
                return FinishReason.CONTENT_FILTER;
            default:
                return FinishReason.OTHER;
        }
    }

    /**
     * Align native rerank scores to input order. The native response is a JSON array of
     * {@code {document, index, score}} objects whose {@code index} is the position in the input
     * documents array; results may arrive in any order, so we place each score at its index.
     *
     * @param json the raw native rerank JSON array
     * @param count the number of input documents (output length)
     * @return scores indexed by input position; positions absent from the response stay {@code 0.0}
     */
    static double[] parseRerankScores(String json, int count) {
        double[] scores = new double[count];
        try {
            JsonNode array = RerankResponseParser.OBJECT_MAPPER.readTree(json);
            if (array.isArray()) {
                int position = 0;
                for (JsonNode entry : array) {
                    // "index" is the input position; fall back to array order when the field is
                    // absent so a response without it never silently yields all-zero scores.
                    int index = entry.path("index").asInt(position);
                    if (index >= 0 && index < count) {
                        scores[index] = entry.path("score").asDouble(0.0);
                    }
                    position++;
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to parse rerank response", e);
        }
        return scores;
    }

    private static net.ladenthin.llama.value.ChatMessage toJllamaMessage(ChatMessage message) {
        switch (message.type()) {
            case SYSTEM:
                return new net.ladenthin.llama.value.ChatMessage(
                        "system", ((SystemMessage) message).text());
            case USER:
                return new net.ladenthin.llama.value.ChatMessage("user", userText((UserMessage) message));
            case AI:
                String aiText = ((AiMessage) message).text();
                return new net.ladenthin.llama.value.ChatMessage(
                        "assistant", aiText == null ? "" : aiText);
            case TOOL_EXECUTION_RESULT:
                ToolExecutionResultMessage tool = (ToolExecutionResultMessage) message;
                return net.ladenthin.llama.value.ChatMessage.toolResult(tool.id(), tool.text());
            default:
                // CUSTOM and any future type: no faithful chat-role mapping exists.
                throw new IllegalArgumentException("Unsupported message type: " + message.type());
        }
    }

    /** Flatten a (possibly multimodal) user message to text; non-text parts (images) are dropped. */
    private static String userText(UserMessage message) {
        if (message.hasSingleText()) {
            return message.singleText();
        }
        StringBuilder text = new StringBuilder();
        for (Content content : message.contents()) {
            if (content.type() == ContentType.TEXT) {
                text.append(((TextContent) content).text());
            }
        }
        return text.toString();
    }

    private static InferenceParameters applySampling(InferenceParameters params, ChatRequest request) {
        if (request.temperature() != null) {
            params = params.withTemperature(request.temperature().floatValue());
        }
        if (request.topP() != null) {
            params = params.withTopP(request.topP().floatValue());
        }
        if (request.topK() != null) {
            params = params.withTopK(request.topK());
        }
        if (request.maxOutputTokens() != null) {
            params = params.withNPredict(request.maxOutputTokens());
        }
        if (request.frequencyPenalty() != null) {
            params = params.withFrequencyPenalty(request.frequencyPenalty().floatValue());
        }
        if (request.presencePenalty() != null) {
            params = params.withPresencePenalty(request.presencePenalty().floatValue());
        }
        List<String> stops = request.stopSequences();
        if (stops != null && !stops.isEmpty()) {
            params = params.withStopStrings(stops.toArray(new String[0]));
        }
        return params;
    }
}
