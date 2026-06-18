// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

/**
 * Optional, self-contained OpenAI-compatible HTTP server built on the in-process
 * {@link net.ladenthin.llama.LlamaModel} API.
 *
 * <p>{@link net.ladenthin.llama.server.LlamaServer} is the {@code main} entry point (and the
 * {@code Main-Class} of the {@code -jar-with-dependencies} assembly). It loads a GGUF model and
 * exposes {@code POST /v1/chat/completions}, {@code POST /v1/completions},
 * {@code POST /v1/embeddings} and {@code GET /v1/models} by forwarding the request body to the
 * matching {@code LlamaModel.handle*} method, which already returns OpenAI-shaped JSON.</p>
 *
 * <p>The HTTP layer is NanoHTTPD (a tiny, dependency-free, Java&nbsp;8 server). The dependency is
 * declared {@code <optional>} so it is bundled in the fat jar but not inherited by library
 * consumers. The routing logic ({@link net.ladenthin.llama.server.OaiRouter}) is decoupled from
 * NanoHTTPD so it can be unit-tested without binding a socket or loading a model.</p>
 *
 * <p>JSpecify {@code @NullMarked} is applied module-wide; everything is non-null unless annotated
 * {@code @Nullable}.</p>
 */
package net.ladenthin.llama.server;
