// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.FinishReason;
import java.util.Objects;
import net.ladenthin.llama.LlamaIterable;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.value.LlamaOutput;

/**
 * langchain4j {@link StreamingChatModel} backed by an in-process java-llama.cpp model.
 *
 * <p>Each generated token is forwarded to {@link StreamingChatResponseHandler#onPartialResponse}; a
 * final {@link StreamingChatResponseHandler#onCompleteResponse} carries the assembled assistant
 * message. Any failure during generation is reported via {@link StreamingChatResponseHandler#onError}.
 *
 * <p>The model is <em>borrowed</em> (never closed here) — see {@link JllamaChatModel}. Tool
 * specifications are <b>not supported on the streaming path yet</b>: a request carrying
 * {@code toolSpecifications()} fails fast with
 * {@link dev.langchain4j.exception.UnsupportedFeatureException} rather than silently generating
 * un-parsed text — use {@link JllamaChatModel} (blocking) for tool calls. JSON
 * {@code responseFormat()} and multimodal user content are forwarded like in the blocking adapter.
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
        if (chatRequest.toolSpecifications() != null
                && !chatRequest.toolSpecifications().isEmpty()) {
            throw new UnsupportedFeatureException(
                    "Tool calling is not supported on the streaming path yet; "
                            + "use JllamaChatModel (blocking) for tool calls");
        }
        StringBuilder full = new StringBuilder();
        try (LlamaIterable stream = model.generateChat(LangChain4jMapping.toStreamingParameters(chatRequest))) {
            for (LlamaOutput output : stream) {
                full.append(output.text);
                handler.onPartialResponse(output.text);
            }
        } catch (Exception e) {
            handler.onError(e);
            return;
        }
        handler.onCompleteResponse(
                ChatResponse.builder()
                        .aiMessage(AiMessage.from(full.toString()))
                        .finishReason(FinishReason.STOP)
                        .build());
    }
}
