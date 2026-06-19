// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.ArrayList;
import java.util.List;
import net.ladenthin.llama.parameters.InferenceParameters;

/**
 * Pure mapping from an OpenAI {@code /v1/chat/completions} request body to {@link InferenceParameters}.
 *
 * <p>The structural fields — {@code messages}, {@code tools}, {@code tool_choice}, and
 * {@code parallel_tool_calls} — are forwarded
 * <em>verbatim</em> as raw JSON so the full OpenAI shape (assistant {@code tool_calls},
 * {@code role:"tool"} results with {@code tool_call_id}, and vision {@code image_url} content parts)
 * round-trips untouched into the native chat-template parser. Sampling fields are translated to the
 * matching {@code InferenceParameters.with*} setter; unknown fields are ignored.
 *
 * <p>The {@code stream} flag is intentionally not mapped here — streaming is selected by the caller
 * ({@link net.ladenthin.llama.LlamaModel#chatComplete} forces it off,
 * {@link net.ladenthin.llama.LlamaModel#streamChatCompletion} forces it on). Stateless and free of JNI
 * and model dependencies, so it is unit-testable with JSON literals alone.
 */
final class OpenAiRequestMapper {

    OpenAiRequestMapper() {}

    /**
     * Translate an OpenAI chat request into {@link InferenceParameters}.
     *
     * @param request the parsed OpenAI request object
     * @return inference parameters carrying the verbatim messages and mapped sampling options
     * @throws IllegalArgumentException if {@code messages} is missing or not a non-empty array
     */
    InferenceParameters toInferenceParameters(JsonNode request) {
        JsonNode messages = request.path("messages");
        if (!messages.isArray() || messages.size() == 0) {
            throw new IllegalArgumentException("'messages' must be a non-empty array");
        }

        // cache_prompt=true reuses the slot's KV prefix across turns — the standard llama.cpp-server
        // default and what IDE clients rely on for acceptable repeated-prefix latency. OpenAI requests
        // never carry this llama.cpp-specific flag, so defaulting it here is safe.
        InferenceParameters params = InferenceParameters.empty()
                .withMessagesJson(messages.toString())
                .withCachePrompt(true);

        JsonNode tools = request.path("tools");
        if (tools.isArray() && tools.size() > 0) {
            params = params.withToolsJson(tools.toString()).withUseChatTemplate(true);
            JsonNode toolChoice = request.path("tool_choice");
            if (toolChoice.isTextual()) {
                params = params.withToolChoice(toolChoice.asText());
            }
            JsonNode parallelToolCalls = request.path("parallel_tool_calls");
            if (parallelToolCalls.isBoolean()) {
                params = params.withParallelToolCalls(parallelToolCalls.asBoolean());
            }
        }

        JsonNode temperature = request.path("temperature");
        if (temperature.isNumber()) {
            params = params.withTemperature((float) temperature.asDouble());
        }
        JsonNode topP = request.path("top_p");
        if (topP.isNumber()) {
            params = params.withTopP((float) topP.asDouble());
        }
        JsonNode topK = request.path("top_k");
        if (topK.isNumber()) {
            params = params.withTopK(topK.asInt());
        }
        JsonNode seed = request.path("seed");
        if (seed.isNumber()) {
            params = params.withSeed(seed.asInt());
        }
        JsonNode presencePenalty = request.path("presence_penalty");
        if (presencePenalty.isNumber()) {
            params = params.withPresencePenalty((float) presencePenalty.asDouble());
        }
        JsonNode frequencyPenalty = request.path("frequency_penalty");
        if (frequencyPenalty.isNumber()) {
            params = params.withFrequencyPenalty((float) frequencyPenalty.asDouble());
        }

        int maxTokens = readMaxTokens(request);
        if (maxTokens > 0) {
            params = params.withNPredict(maxTokens);
        }

        String[] stops = readStops(request);
        if (stops.length > 0) {
            params = params.withStopStrings(stops);
        }

        // Forward stream_options verbatim (e.g. {"include_usage":true}) so the native server emits the
        // trailing usage chunk the OpenAI streaming protocol — and the Copilot custom endpoint — expect.
        JsonNode streamOptions = request.path("stream_options");
        if (streamOptions.isObject()) {
            params = params.withStreamOptions(streamOptions.toString());
        }

        // Forward response_format verbatim (json_object / json_schema) so the native server applies the
        // matching grammar constraint — the OpenAI "structured outputs" feature used by strict clients.
        JsonNode responseFormat = request.path("response_format");
        if (responseFormat.isObject()) {
            params = params.withResponseFormat(responseFormat.toString());
        }

        return params;
    }

    /**
     * Read the output-token cap, preferring the newer {@code max_completion_tokens} over the legacy
     * {@code max_tokens}.
     *
     * @param request the parsed OpenAI request object
     * @return the requested cap, or {@code -1} when neither field is a number
     */
    private int readMaxTokens(JsonNode request) {
        JsonNode maxCompletion = request.path("max_completion_tokens");
        if (maxCompletion.isNumber()) {
            return maxCompletion.asInt();
        }
        JsonNode maxTokens = request.path("max_tokens");
        if (maxTokens.isNumber()) {
            return maxTokens.asInt();
        }
        return -1;
    }

    /**
     * Read the {@code stop} field, which OpenAI permits as either a single string or an array of
     * strings.
     *
     * @param request the parsed OpenAI request object
     * @return the stop strings (possibly empty, never {@code null})
     */
    private String[] readStops(JsonNode request) {
        JsonNode stop = request.path("stop");
        List<String> stops = new ArrayList<>();
        if (stop.isTextual()) {
            stops.add(stop.asText());
        } else if (stop.isArray()) {
            for (JsonNode entry : stop) {
                if (entry.isTextual()) {
                    stops.add(entry.asText());
                }
            }
        }
        return stops.toArray(new String[0]);
    }
}
