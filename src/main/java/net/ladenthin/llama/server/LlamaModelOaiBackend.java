// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.ToString;
import net.ladenthin.llama.LlamaModel;

/**
 * {@link OaiBackend} backed by a loaded {@link LlamaModel}. Each operation forwards the raw request
 * JSON to the matching {@code LlamaModel.handle*} method, which already produces
 * OpenAI-compatible response JSON, so no per-field marshalling happens here.
 *
 * <p>The model is owned by the caller ({@link LlamaServer}); this class neither closes it nor holds
 * any other resource.</p>
 */
@ToString
public final class LlamaModelOaiBackend implements OaiBackend {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private final LlamaModel model;
    private final String modelId;

    /**
     * Create a backend over a loaded model.
     *
     * @param model   the loaded model to serve requests with
     * @param modelId the identifier reported by {@link #listModels()} and echoed in responses
     */
    public LlamaModelOaiBackend(LlamaModel model, String modelId) {
        this.model = model;
        this.modelId = modelId;
    }

    @Override
    public String chatCompletions(String requestJson) {
        return model.handleChatCompletions(requestJson);
    }

    @Override
    public String completions(String requestJson) {
        return model.handleCompletionsOai(requestJson);
    }

    @Override
    public String embeddings(String requestJson) {
        return model.handleEmbeddings(requestJson, true);
    }

    @Override
    public String listModels() {
        final ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("object", "list");
        final ArrayNode data = root.putArray("data");
        final ObjectNode entry = data.addObject();
        entry.put("id", modelId);
        entry.put("object", "model");
        entry.put("owned_by", "llamacpp");
        // ObjectNode.toString() emits valid JSON without a checked exception.
        return root.toString();
    }
}
