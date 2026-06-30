// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

/**
 * langchain4j adapters backed by an in-process java-llama.cpp {@link net.ladenthin.llama.LlamaModel}
 * over JNI — no HTTP server, no separate process.
 *
 * <ul>
 *   <li>{@link net.ladenthin.llama.langchain4j.JllamaChatModel} — {@code ChatModel}</li>
 *   <li>{@link net.ladenthin.llama.langchain4j.JllamaStreamingChatModel} — {@code StreamingChatModel}</li>
 *   <li>{@link net.ladenthin.llama.langchain4j.JllamaEmbeddingModel} — {@code EmbeddingModel}</li>
 *   <li>{@link net.ladenthin.llama.langchain4j.JllamaScoringModel} — {@code ScoringModel} (re-ranking)</li>
 * </ul>
 *
 * <p>Every adapter <em>borrows</em> a model the caller has already loaded and keeps owning: the
 * adapter never loads or closes the native model. This artifact depends on {@code langchain4j-core}
 * but the core {@code net.ladenthin:llama} binding does not depend on langchain4j, so plain
 * java-llama.cpp users never pull langchain4j transitively.
 */
package net.ladenthin.llama.langchain4j;
