// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.parameters.InferenceParameters;

/**
 * Production {@link ChatBackend} that runs requests against a loaded {@link LlamaModel}.
 *
 * <p>Non-streaming requests reuse {@link LlamaModel#chatComplete(InferenceParameters)}, whose return
 * value is already a verbatim OpenAI {@code chat.completion} body. Streaming requests use
 * {@link LlamaModel#streamChatCompletion(InferenceParameters, java.util.function.Consumer)}, which
 * emits OpenAI {@code chat.completion.chunk} objects (including {@code delta.tool_calls}).
 *
 * <p>The streaming sink may fail with {@link IOException} (client disconnect); because the underlying
 * model API takes a {@link java.util.function.Consumer} (no checked exceptions), that failure is
 * relayed across the boundary via {@link java.io.UncheckedIOException} and unwrapped here so the
 * in-flight native task is cancelled.
 */
final class LlamaModelChatBackend implements ChatBackend {

    private final LlamaModel model;
    private final OpenAiRequestMapper mapper;

    /**
     * Create a backend over the given model.
     *
     * @param model the loaded model to run completions against
     * @param mapper the OpenAI-request to {@link InferenceParameters} mapper
     */
    LlamaModelChatBackend(LlamaModel model, OpenAiRequestMapper mapper) {
        this.model = model;
        this.mapper = mapper;
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
}
