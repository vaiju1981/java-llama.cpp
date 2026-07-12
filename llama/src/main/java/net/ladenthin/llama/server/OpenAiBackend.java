// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;

/**
 * The inference engine seam behind {@link OpenAiCompatServer}.
 *
 * <p>Decoupling the HTTP layer from {@link net.ladenthin.llama.LlamaModel} lets the whole server —
 * routing, authentication, Server-Sent-Events framing, heartbeats — be exercised by tests with a fake
 * backend, with no native library and no model loaded. The production implementation is
 * {@link LlamaModelBackend}.
 *
 * <p>Every method receives the parsed OpenAI request object (already validated as a JSON object by the
 * handler) and returns the OpenAI-shaped response JSON, except {@link #stream} which delivers chunks
 * incrementally. The {@code GET /v1/models} response is built from configuration alone and so is not
 * part of this seam.
 */
interface OpenAiBackend {

    /**
     * Return llama.cpp server metrics, including per-slot cache counters.
     * Test backends may rely on the empty default.
     *
     * @return metrics JSON
     * @throws IOException if metrics cannot be read
     */
    default String metrics() throws IOException {
        return "{\"slots\":[]}";
    }

    /**
     * Run a non-streaming chat completion ({@code POST /v1/chat/completions}).
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

    /**
     * Run a (non-streaming) text completion ({@code POST /v1/completions}). The request body is
     * forwarded verbatim to the native OpenAI-compatible completion handler.
     *
     * @param request the parsed OpenAI {@code /v1/completions} request (must contain {@code "prompt"})
     * @return the OpenAI {@code text_completion} response serialized as JSON
     * @throws IOException if generation fails in a way the caller should surface as a server error
     */
    String completions(JsonNode request) throws IOException;

    /**
     * Run a <em>streaming</em> text completion ({@code POST /v1/completions} with {@code stream:true}),
     * delivering each OpenAI {@code text_completion} chunk to {@code sink} in order. Implementations must
     * not emit the terminating {@code [DONE]} marker; the caller adds it. The default throws
     * {@link UnsupportedOperationException}; backends that support streaming completions override it.
     *
     * @param request the parsed OpenAI {@code /v1/completions} request (must contain {@code "prompt"})
     * @param sink receiver for each streamed chunk's JSON
     * @throws IOException if a chunk cannot be delivered or generation fails
     */
    default void streamCompletions(JsonNode request, ChunkSink sink) throws IOException {
        throw new UnsupportedOperationException("streaming /v1/completions is not supported by this backend");
    }

    /**
     * Generate embeddings ({@code POST /v1/embeddings}). Requires the model to have been loaded in
     * embedding mode; otherwise the native call fails and the caller surfaces a server error.
     *
     * @param request the parsed OpenAI {@code /v1/embeddings} request (must contain {@code "input"})
     * @return the OpenAI embeddings response serialized as JSON
     * @throws IOException if generation fails in a way the caller should surface as a server error
     */
    String embeddings(JsonNode request) throws IOException;

    /**
     * Rerank documents against a query ({@code POST /v1/rerank}). Requires the model to have been loaded
     * in reranking mode; otherwise the native call fails and the caller surfaces a server error.
     *
     * @param request the parsed rerank request ({@code query} string + {@code documents} array, optional
     *                {@code top_n})
     * @return the rerank response serialized as JSON ({@code results}/{@code data} of
     *         {@code {index, relevance_score}})
     * @throws IOException if reranking fails in a way the caller should surface as a server error
     */
    String rerank(JsonNode request) throws IOException;

    /**
     * Run a (non-streaming) fill-in-the-middle completion ({@code POST /infill}). The request body is
     * forwarded verbatim to the native llama.cpp infill handler, which applies the model's FIM control
     * tokens server-side from GGUF metadata — so callers send raw {@code input_prefix} /
     * {@code input_suffix} (and optional {@code input_extra} / {@code prompt}). This is the endpoint
     * that drives local ghost-text autocomplete clients (llama.vscode, llama.vim, Twinny, Tabby,
     * Continue's {@code llama.cpp} provider).
     *
     * @param request the parsed llama.cpp {@code /infill} request (typically {@code input_prefix} +
     *                {@code input_suffix})
     * @return the infill response serialized as JSON (clients read the {@code "content"} field)
     * @throws IOException if generation fails in a way the caller should surface as a server error
     */
    String infill(JsonNode request) throws IOException;

    /**
     * Tokenize text to token ids ({@code POST /tokenize}). The request carries {@code content}
     * (string), optional {@code add_special} (boolean, default true) and {@code with_pieces}
     * (boolean, default false). The native call returns the upstream llama.cpp {@code /tokenize}
     * JSON (a {@code tokens} array plus optional piece strings) verbatim.
     *
     * @param request the parsed {@code /tokenize} request
     * @return the tokenize response serialized as JSON
     * @throws IOException if tokenization fails
     */
    default String tokenize(JsonNode request) throws IOException {
        throw new UnsupportedOperationException("/tokenize is not supported by this backend");
    }

    /**
     * Detokenize token ids back to text ({@code POST /detokenize}). The request carries
     * {@code tokens} (array of integer ids); the native call returns the upstream llama.cpp
     * {@code /detokenize} JSON verbatim.
     *
     * @param request the parsed {@code /detokenize} request
     * @return the detokenize response serialized as JSON
     * @throws IOException if detokenization fails
     */
    default String detokenize(JsonNode request) throws IOException {
        throw new UnsupportedOperationException("/detokenize is not supported by this backend");
    }

    /**
     * Apply the model's chat template to a request ({@code POST /apply-template}). The request body
     * is the full OpenAI-style parameter blob; the native call returns the templated prompt string,
     * wrapped as {@code {"content": "..."}} for a stable JSON contract.
     *
     * @param request the parsed {@code /apply-template} request
     * @return the templated prompt as {@code {"content": "..."}} JSON
     * @throws IOException if the template cannot be applied
     */
    default String applyTemplate(JsonNode request) throws IOException {
        throw new UnsupportedOperationException("/apply-template is not supported by this backend");
    }

    /**
     * List the LoRA adapters currently applied to the model ({@code GET /lora-adapters}).
     * Returns the upstream llama.cpp {@code /lora-adapters} JSON verbatim.
     *
     * @return the LoRA adapter list serialized as JSON
     * @throws IOException if the adapter list cannot be read
     */
    default String loraAdapters() throws IOException {
        throw new UnsupportedOperationException("/lora-adapters is not supported by this backend");
    }

    /**
     * Apply a set of LoRA adapters ({@code POST /lora-adapters}). The request carries an
     * {@code adapters} array of {@code {id, scale}} objects; each is forwarded to
     * {@link net.ladenthin.llama.LlamaModel#setLoraAdapter(int, float)}.
     *
     * @param request the parsed {@code /lora-adapters} request
     * @return a {@code {"status":"ok"}} JSON acknowledgement
     * @throws IOException if the adapters cannot be applied
     */
    default String setLoraAdapters(JsonNode request) throws IOException {
        throw new UnsupportedOperationException("/lora-adapters is not supported by this backend");
    }

    /**
     * Count the tokens a request would consume ({@code POST /count_tokens}, plus the per-API
     * aliases {@code /v1/chat/completions/input_tokens},
     * {@code /v1/responses/input_tokens} and {@code /v1/messages/count_tokens}). The text is
     * taken from {@code messages} (concatenated text content), otherwise {@code prompt} /
     * {@code input}; the native tokenizer returns the count, wrapped as {@code {"count": N}}.
     *
     * @param request the parsed count-tokens request
     * @return the token count as {@code {"count": N}} JSON
     * @throws IOException if tokenization fails
     */
    default String countTokens(JsonNode request) throws IOException {
        throw new UnsupportedOperationException("/count_tokens is not supported by this backend");
    }

    /**
     * Transcribe an audio file to text ({@code POST /v1/audio/transcriptions}). The upstream llama.cpp
     * endpoint expects {@code multipart/form-data} (an audio file plus optional parameters such as
     * {@code language} and {@code temperature}); the backend parses the multipart payload itself. Native
     * Whisper support is gated on the {@code GIT_TAG} bump in the serving-server plan, so the default
     * throws.
     *
     * @param request the parsed request object (a JSON object in tests; the production body is multipart)
     * @return the transcription, typically {@code {"text": "..."}}
     * @throws IOException if transcription fails
     */
    default String audioTranscriptions(JsonNode request) throws IOException {
        throw new UnsupportedOperationException("/v1/audio/transcriptions is not supported by this backend");
    }

    /**
     * Real-time control channel ({@code POST /v1/chat/completions/control}): pause, resume, reset, abort
     * or reconfigure an in-flight task. Native support is gated on the {@code GIT_TAG} bump in the
     * serving-server plan, so the default throws.
     *
     * @param request the control request ({@code action}, optional {@code id} / {@code data})
     * @return the control acknowledgement JSON
     * @throws IOException if the control request cannot be applied
     */
    default String control(JsonNode request) throws IOException {
        throw new UnsupportedOperationException("/v1/chat/completions/control is not supported by this backend");
    }

    /**
     * List the active streaming conversation handles ({@code GET /v1/streams/lookup}).
     *
     * @return the active-stream list as JSON
     * @throws IOException if the registry cannot be read
     */
    default String streamsLookup() throws IOException {
        throw new UnsupportedOperationException("/v1/streams/lookup is not supported by this backend");
    }

    /**
     * Fetch the partial output of an in-flight stream ({@code GET /v1/stream/:conv_id}).
     *
     * @param convId the conversation id of the stream to read
     * @return the stream's partial output as JSON
     * @throws IOException if the stream cannot be read
     */
    default String streamGet(String convId) throws IOException {
        throw new UnsupportedOperationException("/v1/stream/:conv_id is not supported by this backend");
    }

    /**
     * Cancel and forget an in-flight stream ({@code DELETE /v1/stream/:conv_id}).
     *
     * @param convId the conversation id of the stream to cancel
     * @return the cancellation acknowledgement JSON
     * @throws IOException if the stream cannot be cancelled
     */
    default String streamDelete(String convId) throws IOException {
        throw new UnsupportedOperationException("/v1/stream/:conv_id DELETE is not supported by this backend");
    }
}
