// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;

/**
 * The chat engine seam behind {@link OpenAiCompatServer}.
 *
 * <p>Decoupling the HTTP layer from {@link net.ladenthin.llama.LlamaModel} lets the whole server —
 * routing, authentication, Server-Sent-Events framing, heartbeats — be exercised by tests with a fake
 * backend, with no native library and no model loaded. The production implementation is
 * {@link LlamaModelChatBackend}.
 *
 * <p>Both methods receive the parsed OpenAI request object (already validated as JSON by the handler).
 */
interface ChatBackend {

    /**
     * Run a non-streaming chat completion.
     *
     * @param request the parsed OpenAI {@code /v1/chat/completions} request
     * @return the complete OpenAI {@code chat.completion} response serialized as JSON
     * @throws IOException if generation fails in a way the caller should surface as a server error
     */
    String complete(JsonNode request) throws IOException;

    /**
     * Run a streaming chat completion, delivering each {@code chat.completion.chunk} to {@code sink}
     * in order. Implementations must not emit the terminating {@code [DONE]} marker; the caller adds it.
     *
     * @param request the parsed OpenAI {@code /v1/chat/completions} request
     * @param sink receiver for each streamed chunk's JSON
     * @throws IOException if a chunk cannot be delivered or generation fails
     */
    void stream(JsonNode request, ChunkSink sink) throws IOException;
}
