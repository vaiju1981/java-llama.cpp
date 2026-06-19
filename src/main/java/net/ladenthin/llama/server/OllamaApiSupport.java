// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.time.Instant;
import org.jspecify.annotations.Nullable;

/**
 * Pure translators between Ollama's native HTTP API and the OpenAI chat shape used internally, so the
 * server can present an Ollama-compatible surface (which Copilot's built-in <em>Ollama</em> provider and
 * Ollama-hardcoded tools target) without a second inference path.
 *
 * <p>Covers the discovery endpoints ({@code /api/version}, {@code /api/tags}, {@code /api/show}) and the
 * {@code /api/chat} request/response translation, including the NDJSON streaming shape (one JSON object
 * per line, terminated by a {@code "done":true} line). Tool-call {@code arguments} are objects in Ollama
 * but JSON-encoded strings in OpenAI, so they are converted on the way in and out.
 *
 * <p>Stateless and free of JNI / model dependencies; unit-testable with JSON literals.
 */
final class OllamaApiSupport {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /** Advertised Ollama API version (clients only check that {@code /api/version} responds with one). */
    static final String OLLAMA_VERSION = "0.1.0";

    private OllamaApiSupport() {}

    private static String nowIso() {
        return Instant.now().toString();
    }

    /**
     * The {@code GET /api/version} body.
     *
     * @return {@code {"version":"<v>"}}
     */
    static String versionJson() {
        return OBJECT_MAPPER.createObjectNode().put("version", OLLAMA_VERSION).toString();
    }

    /**
     * The {@code GET /api/tags} body advertising the single served model.
     *
     * @param modelId the model id
     * @return an Ollama model-list object serialized as JSON
     */
    static String tagsJson(String modelId) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        ArrayNode models = root.putArray("models");
        ObjectNode model = models.addObject();
        model.put("name", modelId);
        model.put("model", modelId);
        model.put("modified_at", nowIso());
        model.put("size", 0L);
        model.put("digest", "");
        ObjectNode details = model.putObject("details");
        details.put("family", "llama");
        details.put("parameter_size", "");
        details.put("quantization_level", "");
        return root.toString();
    }

    /**
     * The {@code POST /api/show} body advertising the model's capabilities and context length — the
     * fields Copilot's Ollama provider reads to enable tools/vision and size prompts.
     *
     * @param modelId the model id
     * @param contextLength the advertised context window
     * @param vision whether image input is supported
     * @return an Ollama show object serialized as JSON
     */
    static String showJson(String modelId, int contextLength, boolean vision) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("license", "");
        root.put("modelfile", "");
        root.put("parameters", "");
        root.put("template", "");
        ObjectNode details = root.putObject("details");
        details.put("family", "llama");
        details.putArray("families").add("llama");
        details.put("parameter_size", "");
        details.put("quantization_level", "");
        ObjectNode modelInfo = root.putObject("model_info");
        modelInfo.put("general.architecture", "llama");
        modelInfo.put("llama.context_length", contextLength);
        ArrayNode capabilities = root.putArray("capabilities");
        capabilities.add("completion");
        capabilities.add("tools");
        capabilities.add("insert"); // fill-in-the-middle via /infill
        if (vision) {
            capabilities.add("vision");
        }
        root.put("model", modelId);
        return root.toString();
    }

    /**
     * Whether the Ollama request asks for a streamed response. Ollama defaults {@code stream} to
     * {@code true} when the field is absent.
     *
     * @param ollamaRequest the parsed Ollama request
     * @return {@code true} unless {@code "stream"} is explicitly {@code false}
     */
    static boolean isStreaming(JsonNode ollamaRequest) {
        JsonNode stream = ollamaRequest.path("stream");
        return !stream.isBoolean() || stream.asBoolean();
    }

    /**
     * Translate an Ollama {@code /api/chat} request into the internal OpenAI chat request shape.
     *
     * @param ollamaRequest the parsed Ollama request
     * @return an OpenAI {@code /v1/chat/completions} request object
     */
    static ObjectNode toOpenAiChatRequest(JsonNode ollamaRequest) {
        ObjectNode openAi = OBJECT_MAPPER.createObjectNode();
        if (ollamaRequest.path("model").isTextual()) {
            openAi.put("model", ollamaRequest.path("model").asText());
        }

        // Messages: copy through, converting any assistant tool_calls.arguments object to the OpenAI
        // JSON-encoded string form.
        ArrayNode messages = openAi.putArray("messages");
        for (JsonNode message : ollamaRequest.path("messages")) {
            messages.add(toOpenAiMessage(message));
        }

        JsonNode tools = ollamaRequest.path("tools");
        if (tools.isArray() && tools.size() > 0) {
            openAi.set("tools", tools.deepCopy());
        }

        // Ollama nests sampling under "options"; map the common knobs onto OpenAI top-level fields.
        JsonNode options = ollamaRequest.path("options");
        copyNumber(options, "temperature", openAi, "temperature");
        copyNumber(options, "top_p", openAi, "top_p");
        copyNumber(options, "top_k", openAi, "top_k");
        copyNumber(options, "seed", openAi, "seed");
        copyNumber(options, "num_predict", openAi, "max_tokens");
        if (options.path("stop").isArray()) {
            openAi.set("stop", options.path("stop").deepCopy());
        }

        // Ollama "format": "json" or a JSON schema → OpenAI response_format.
        JsonNode format = ollamaRequest.path("format");
        if (format.isTextual() && "json".equals(format.asText())) {
            openAi.putObject("response_format").put("type", "json_object");
        } else if (format.isObject()) {
            ObjectNode responseFormat = openAi.putObject("response_format");
            responseFormat.put("type", "json_schema");
            responseFormat.putObject("json_schema").set("schema", format.deepCopy());
        }
        return openAi;
    }

    private static ObjectNode toOpenAiMessage(JsonNode ollamaMessage) {
        ObjectNode message = ollamaMessage.deepCopy();
        JsonNode toolCalls = message.path("tool_calls");
        if (toolCalls.isArray()) {
            for (JsonNode toolCall : toolCalls) {
                JsonNode arguments = toolCall.path("function").path("arguments");
                if (arguments.isObject() || arguments.isArray()) {
                    ((ObjectNode) toolCall.path("function")).put("arguments", arguments.toString());
                }
            }
        }
        return message;
    }

    private static void copyNumber(JsonNode from, String fromKey, ObjectNode to, String toKey) {
        JsonNode value = from.path(fromKey);
        if (value.isNumber()) {
            to.set(toKey, value);
        }
    }

    /**
     * Translate a non-streaming OpenAI {@code chat.completion} into an Ollama {@code /api/chat} response.
     *
     * @param openAiCompletionJson the OpenAI completion body
     * @param model the model id to echo
     * @return the Ollama chat response serialized as JSON
     */
    static String toOllamaChatResponse(String openAiCompletionJson, String model) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("model", model);
        root.put("created_at", nowIso());
        ObjectNode message = root.putObject("message");
        message.put("role", "assistant");
        message.put("content", "");
        String doneReason = "stop";
        try {
            JsonNode completion = OBJECT_MAPPER.readTree(openAiCompletionJson);
            JsonNode choice = completion.path("choices").path(0);
            JsonNode openAiMessage = choice.path("message");
            message.put("content", openAiMessage.path("content").asText(""));
            ArrayNode ollamaToolCalls = toOllamaToolCalls(openAiMessage.path("tool_calls"));
            if (ollamaToolCalls.size() > 0) {
                message.set("tool_calls", ollamaToolCalls);
            }
            if (choice.path("finish_reason").isTextual()) {
                doneReason = choice.path("finish_reason").asText();
            }
            JsonNode usage = completion.path("usage");
            if (usage.isObject()) {
                root.put("prompt_eval_count", usage.path("prompt_tokens").asInt(0));
                root.put("eval_count", usage.path("completion_tokens").asInt(0));
            }
        } catch (IOException e) {
            // Defensive: an unexpected body still yields a valid, empty Ollama "done" response.
            doneReason = "stop";
        }
        root.put("done", true);
        root.put("done_reason", doneReason);
        return root.toString();
    }

    /**
     * Translate one streamed OpenAI chunk into an Ollama NDJSON content line, or return {@code null} when
     * the chunk carries no assistant text to emit (role-only, finish-only or usage-only chunks).
     *
     * @param openAiChunkJson one OpenAI {@code chat.completion.chunk}
     * @param model the model id to echo
     * @return the Ollama NDJSON line (with trailing newline), or {@code null} if nothing to emit
     */
    static @Nullable String toOllamaContentLine(String openAiChunkJson, String model) {
        try {
            JsonNode chunk = OBJECT_MAPPER.readTree(openAiChunkJson);
            JsonNode content = chunk.path("choices").path(0).path("delta").path("content");
            if (!content.isTextual() || content.asText().isEmpty()) {
                return null;
            }
            ObjectNode root = OBJECT_MAPPER.createObjectNode();
            root.put("model", model);
            root.put("created_at", nowIso());
            ObjectNode message = root.putObject("message");
            message.put("role", "assistant");
            message.put("content", content.asText());
            root.put("done", false);
            return root.toString() + "\n";
        } catch (IOException e) {
            return null;
        }
    }

    /**
     * Build the terminating Ollama NDJSON line ({@code "done":true}), attaching any tool calls
     * reconstructed from the stream.
     *
     * @param model the model id to echo
     * @param accumulator the tool-call accumulator fed with the stream's chunks
     * @return the final Ollama NDJSON line (with trailing newline)
     */
    static String toOllamaDoneLine(String model, ToolCallDeltaAccumulator accumulator) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("model", model);
        root.put("created_at", nowIso());
        ObjectNode message = root.putObject("message");
        message.put("role", "assistant");
        message.put("content", "");
        if (accumulator.hasToolCalls()) {
            ArrayNode ollamaToolCalls = toOllamaToolCalls(accumulator.toOpenAiToolCalls());
            if (ollamaToolCalls.size() > 0) {
                message.set("tool_calls", ollamaToolCalls);
            }
        }
        root.put("done", true);
        root.put("done_reason", "stop");
        return root.toString() + "\n";
    }

    /** Convert OpenAI tool calls (arguments = JSON string) to Ollama tool calls (arguments = object). */
    private static ArrayNode toOllamaToolCalls(JsonNode openAiToolCalls) {
        ArrayNode out = OBJECT_MAPPER.createArrayNode();
        if (!openAiToolCalls.isArray()) {
            return out;
        }
        for (JsonNode openAiToolCall : openAiToolCalls) {
            JsonNode function = openAiToolCall.path("function");
            ObjectNode ollamaToolCall = out.addObject();
            ObjectNode ollamaFunction = ollamaToolCall.putObject("function");
            ollamaFunction.put("name", function.path("name").asText(""));
            ollamaFunction.set("arguments", parseArgumentsToObject(function.path("arguments")));
        }
        return out;
    }

    private static JsonNode parseArgumentsToObject(JsonNode arguments) {
        if (arguments.isObject() || arguments.isArray()) {
            return arguments;
        }
        if (arguments.isTextual()) {
            try {
                return OBJECT_MAPPER.readTree(arguments.asText());
            } catch (IOException e) {
                // Fall through to an empty object on unparseable arguments.
                return OBJECT_MAPPER.createObjectNode();
            }
        }
        return OBJECT_MAPPER.createObjectNode();
    }
}
