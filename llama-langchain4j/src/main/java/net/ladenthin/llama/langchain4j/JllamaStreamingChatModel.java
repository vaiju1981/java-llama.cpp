// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.Objects;
import net.ladenthin.llama.LlamaModel;

/**
 * langchain4j {@link StreamingChatModel} backed by an in-process java-llama.cpp model.
 *
 * <p>Streams over the native OpenAI {@code chat.completion.chunk} path
 * ({@code LlamaModel.streamChatCompletion}), so the full event surface is forwarded:
 * assistant text via {@link StreamingChatResponseHandler#onPartialResponse(String)}, reasoning
 * deltas via {@link StreamingChatResponseHandler#onPartialThinking}, and <b>streamed tool
 * calls</b> — {@code delta.tool_calls} fragments are forwarded as
 * {@link StreamingChatResponseHandler#onPartialToolCall} events, completed via
 * {@link StreamingChatResponseHandler#onCompleteToolCall}, and carried on the final
 * {@link ChatResponse} as {@code AiMessage.toolExecutionRequests()} (finish reason
 * {@code TOOL_EXECUTION}). JSON {@code responseFormat()} and multimodal user content are
 * forwarded like in the blocking adapter. Any failure during generation is reported via
 * {@link StreamingChatResponseHandler#onError}.
 *
 * <p>The model is <em>borrowed</em> (never closed here) — see {@link JllamaChatModel}.
 */
public final class JllamaStreamingChatModel implements StreamingChatModel {

    private final LlamaModel model;

    /**
     * Creates a streaming chat model over a borrowed {@link LlamaModel}.
     *
     * @param model the loaded model to drive; not closed by this adapter
     */
    public JllamaStreamingChatModel(LlamaModel model) {
        this.model = Objects.requireNonNull(model, "model");
    }

    @Override
    public void doChat(ChatRequest chatRequest, StreamingChatResponseHandler handler) {
        StreamingChunkAssembler assembler = new StreamingChunkAssembler(handler);
        try {
            model.streamChatCompletion(LangChain4jMapping.toStreamingParameters(chatRequest), assembler::accept);
        } catch (Exception e) {
            handler.onError(e);
            return;
        }
        handler.onCompleteResponse(assembler.complete());
    }
}
