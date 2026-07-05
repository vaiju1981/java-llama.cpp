// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.util.Objects;
import net.ladenthin.llama.LlamaModel;

/**
 * langchain4j {@link ChatModel} backed by an in-process java-llama.cpp model (over JNI, no HTTP).
 *
 * <p>The model is <em>borrowed</em>: this adapter never loads or closes it. Construct it from a
 * {@link LlamaModel} you already own and keep managing that model's lifecycle (try-with-resources or
 * an explicit {@code close()}). One {@code LlamaModel} can back several adapters at once.
 *
 * <p>Mapped today: messages (system/user/assistant-with-tool-calls/tool-result), multimodal user
 * content (text + image/audio parts, routed through the mtmd pipeline when the model was loaded
 * with an {@code mmproj}), tool specifications ({@code toolSpecifications()} + {@code toolChoice()}
 * — a response with {@code tool_calls} comes back as {@link dev.langchain4j.data.message.AiMessage
 * AiMessage}{@code .toolExecutionRequests()} with finish reason {@code TOOL_EXECUTION}), JSON
 * {@code responseFormat()} (plain JSON mode and JSON-schema-constrained structured output), and the
 * sampling parameters {@code temperature}/{@code topP}/{@code topK}/{@code maxOutputTokens}/{@code
 * frequencyPenalty}/{@code presencePenalty}/{@code stopSequences}.
 */
public final class JllamaChatModel implements ChatModel {

    private final LlamaModel model;

    /**
     * Creates a chat model over a borrowed {@link LlamaModel}.
     *
     * @param model the loaded model to drive; not closed by this adapter
     */
    public JllamaChatModel(LlamaModel model) {
        this.model = Objects.requireNonNull(model, "model");
    }

    @Override
    public ChatResponse doChat(ChatRequest chatRequest) {
        net.ladenthin.llama.value.ChatResponse response =
                model.chat(LangChain4jMapping.toJllamaRequest(chatRequest));
        return LangChain4jMapping.toLangChainResponse(response);
    }
}
