// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.util.UUID;
import net.ladenthin.llama.LlamaIterable;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.value.LlamaOutput;
import net.ladenthin.llama.value.StopReason;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Production {@link OpenAiBackend} that runs requests against a loaded {@link LlamaModel}.
 *
 * <p>Non-streaming chat reuses {@link LlamaModel#chatComplete(InferenceParameters)}, whose return value
 * is already a verbatim OpenAI {@code chat.completion} body. Streaming chat uses
 * {@link LlamaModel#streamChatCompletion(InferenceParameters, java.util.function.Consumer)}, which emits
 * OpenAI {@code chat.completion.chunk} objects (including {@code delta.tool_calls}). Text completions and
 * embeddings forward the request body verbatim to {@link LlamaModel#handleCompletionsOai(String)} /
 * {@link LlamaModel#handleEmbeddings(String, boolean)}, which already return OpenAI-shaped JSON.
 *
 * <p>The streaming sink may fail with {@link IOException} (client disconnect); because the underlying
 * model API takes a {@link java.util.function.Consumer} (no checked exceptions), that failure is
 * relayed across the boundary via {@link java.io.UncheckedIOException} and unwrapped here so the
 * in-flight native task is cancelled.
 */
final class LlamaModelBackend implements OpenAiBackend {

    private static final Logger LOG = LoggerFactory.getLogger(LlamaModelBackend.class);

    private final LlamaModel model;
    private final OpenAiRequestMapper mapper;
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /**
     * Create a backend over the given model.
     *
     * @param model the loaded model to run completions against
     * @param mapper the OpenAI-request to {@link InferenceParameters} mapper
     */
    LlamaModelBackend(LlamaModel model, OpenAiRequestMapper mapper) {
        this.model = model;
        this.mapper = mapper;
    }

    @Override
    public String metrics() {
        return model.getMetrics();
    }

    @Override
    public String complete(JsonNode request) {
        return model.chatComplete(mapper.toInferenceParameters(request));
    }

    @Override
    public void stream(JsonNode request, ChunkSink sink) throws IOException {
        InferenceParameters params = mapper.toInferenceParameters(request);
        // Holds an IOException thrown by the sink so it can be rethrown after the model API (which
        // only understands unchecked exceptions) unwinds and cancels the native task.
        final IOException[] sinkFailure = new IOException[1];
        try {
            model.streamChatCompletion(params, chunkJson -> {
                try {
                    sink.accept(chunkJson);
                } catch (IOException e) {
                    sinkFailure[0] = e;
                    throw new java.io.UncheckedIOException(e);
                }
            });
        } catch (java.io.UncheckedIOException e) {
            IOException cause = sinkFailure[0];
            if (cause != null) {
                throw cause;
            }
            throw e;
        }
    }

    @Override
    public String completions(JsonNode request) {
        // The native /v1/completions handler parses the OpenAI body itself; forward it verbatim.
        return model.handleCompletionsOai(request.toString());
    }

    @Override
    public void streamCompletions(JsonNode request, ChunkSink sink) throws IOException {
        InferenceParameters params = mapper.toCompletionParameters(request);
        String modelId = request.path("model").asText("llama");
        String id = "cmpl-" + UUID.randomUUID().toString().replace("-", "");
        long created = System.currentTimeMillis() / 1000L;
        // Relays a sink IOException (client disconnect) out of the token loop; try-with-resources then
        // cancels the in-flight native task via LlamaIterable.close().
        IOException sinkFailure = null;
        try (LlamaIterable it = model.generate(params)) {
            for (LlamaOutput out : it) {
                String finishReason = out.stop ? completionFinishReason(out.stopReason) : null;
                try {
                    sink.accept(OpenAiSseFormatter.completionChunk(id, created, modelId, out.text, finishReason));
                } catch (IOException e) {
                    sinkFailure = e;
                    break;
                }
            }
        }
        if (sinkFailure != null) {
            throw sinkFailure;
        }
    }

    /** Map a {@link StopReason} to the OpenAI {@code finish_reason} ("length" on the token cap, else "stop"). */
    private static String completionFinishReason(StopReason reason) {
        return reason == StopReason.MAX_TOKENS ? "length" : "stop";
    }

    @Override
    public String embeddings(JsonNode request) {
        // oaiCompat=true so the response uses the OpenAI {"object":"list","data":[{embedding}]} shape.
        return model.handleEmbeddings(request.toString(), true);
    }

    @Override
    public String infill(JsonNode request) {
        // The native /infill handler parses the body itself (input_prefix/input_suffix/...) and applies
        // the model's FIM tokens from GGUF metadata; forward verbatim.
        return model.handleInfill(request.toString());
    }

    @Override
    public String tokenize(JsonNode request) {
        String content = request.path("content").asText("");
        boolean addSpecial = request.path("add_special").asBoolean(true);
        boolean withPieces = request.path("with_pieces").asBoolean(false);
        return model.handleTokenize(content, addSpecial, withPieces);
    }

    @Override
    public String detokenize(JsonNode request) {
        JsonNode tokens = request.path("tokens");
        int[] ids = new int[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            ids[i] = tokens.get(i).asInt();
        }
        return model.handleDetokenize(ids);
    }

    @Override
    public String applyTemplate(JsonNode request) {
        // The native call returns the raw templated prompt string; wrap it so the HTTP
        // contract is stable JSON rather than bare text.
        ObjectNode response = OBJECT_MAPPER.createObjectNode();
        response.put("content", model.applyTemplate(request.toString()));
        return response.toString();
    }

    @Override
    public String loraAdapters() {
        return model.getLoraAdaptersJson();
    }

    @Override
    public String setLoraAdapters(JsonNode request) {
        JsonNode adapters = request.path("adapters");
        for (JsonNode adapter : adapters) {
            int id = adapter.path("id").asInt();
            float scale = (float) adapter.path("scale").asDouble(1.0);
            model.setLoraAdapter(id, scale);
        }
        ObjectNode response = OBJECT_MAPPER.createObjectNode();
        response.put("status", "ok");
        return response.toString();
    }

    @Override
    public String countTokens(JsonNode request) {
        StringBuilder text = new StringBuilder();
        JsonNode messages = request.path("messages");
        if (messages.isArray()) {
            for (JsonNode message : messages) {
                JsonNode content = message.path("content");
                if (content.isTextual()) {
                    text.append(content.asText());
                }
            }
        } else {
            String prompt = request.path("prompt").asText("");
            if (!prompt.isEmpty()) {
                text.append(prompt);
            } else {
                text.append(request.path("input").asText(""));
            }
        }
        // Native tokenizer returns a {"tokens":[...]} JSON; count its array length.
        int count = 0;
        try {
            JsonNode tokenized = OBJECT_MAPPER.readTree(model.handleTokenize(text.toString(), true, false));
            JsonNode tokens = tokenized.path("tokens");
            if (tokens.isArray()) {
                count = tokens.size();
            }
        } catch (IOException | RuntimeException e) {
            LOG.warn("count_tokens tokenization failed", e);
        }
        ObjectNode response = OBJECT_MAPPER.createObjectNode();
        response.put("count", count);
        return response.toString();
    }

    @Override
    public String rerank(JsonNode request) {
        final String query = OaiRerankSupport.readQuery(request);
        final String[] documents = OaiRerankSupport.readDocuments(request);
        final int topN = OaiRerankSupport.readTopN(request);
        final String requestModel = request.path("model").asText("");
        final String nativeJson = model.handleRerank(query, documents);
        return OaiRerankSupport.toOaiResponse(nativeJson, requestModel, topN);
    }

    @Override
    public String saveSlots(JsonNode request) {
        int slotId = request.path("slot_id").asInt(-1);
        String filename = request.path("filename").asText("");
        return model.saveSlot(slotId, filename.isEmpty() ? null : filename);
    }

    @Override
    public String eraseSlot(int id) {
        return model.eraseSlot(id);
    }
}
