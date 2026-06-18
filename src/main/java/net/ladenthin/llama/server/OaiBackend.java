// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

/**
 * The inference operations the {@link OaiRouter} forwards HTTP requests to, abstracted behind an
 * interface so the router can be unit-tested without loading a native model. The production
 * implementation is {@link LlamaModelOaiBackend}, which delegates to
 * {@link net.ladenthin.llama.LlamaModel}.
 *
 * <p>Each request method takes the raw OpenAI-compatible request body and returns the raw
 * OpenAI-compatible response JSON. Implementations may throw a {@link RuntimeException} (e.g.
 * {@link net.ladenthin.llama.exception.LlamaException}) on inference failure; the router converts
 * that into an HTTP {@code 500} error response.</p>
 */
public interface OaiBackend {

    /**
     * Run a chat completion ({@code POST /v1/chat/completions}).
     *
     * @param requestJson the OAI chat-completion request body (must contain {@code "messages"})
     * @return the OAI chat-completion response JSON
     */
    String chatCompletions(String requestJson);

    /**
     * Run a text completion ({@code POST /v1/completions}).
     *
     * @param requestJson the OAI completion request body (must contain {@code "prompt"})
     * @return the OAI completion response JSON
     */
    String completions(String requestJson);

    /**
     * Generate embeddings ({@code POST /v1/embeddings}).
     *
     * @param requestJson the OAI embeddings request body (must contain {@code "input"})
     * @return the OAI embeddings response JSON
     */
    String embeddings(String requestJson);

    /**
     * List the available model(s) ({@code GET /v1/models}).
     *
     * @return the OAI model-list response JSON
     */
    String listModels();
}
