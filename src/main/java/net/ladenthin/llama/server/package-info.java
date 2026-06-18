// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

/**
 * Optional OpenAI-compatible HTTP server over a loaded {@link net.ladenthin.llama.LlamaModel}.
 *
 * <p><strong>Interim state — two implementations pending consolidation.</strong> This package
 * currently contains <em>two</em> independent OpenAI-compatible server implementations that landed
 * on separate branches and are awaiting a "best of both" merge (tracked in {@code TODO.md}). Both
 * let editors and tools that speak the OpenAI Chat Completions protocol (for example a VS&nbsp;Code
 * Copilot "Custom Endpoint") drive a local GGUF model running in-process through the JNI binding,
 * and both are faithful pass-throughs that do not implement or execute tools themselves.</p>
 *
 * <ul>
 *   <li>{@link net.ladenthin.llama.server.LlamaServer} /
 *       {@link net.ladenthin.llama.server.OaiHttpServer} — a NanoHTTPD-based server.
 *       {@code LlamaServer} is a {@code main} entry point (and the {@code Main-Class} of the
 *       {@code -jar-with-dependencies} assembly). It exposes {@code POST /v1/chat/completions},
 *       {@code POST /v1/completions}, {@code POST /v1/embeddings} and {@code GET /v1/models} by
 *       forwarding the request body to the matching {@code LlamaModel.handle*} method, which already
 *       returns OpenAI-shaped JSON. Routing ({@link net.ladenthin.llama.server.OaiRouter}) is
 *       decoupled from NanoHTTPD so it is unit-testable without binding a socket or loading a model.
 *       NanoHTTPD is an {@code <optional>} dependency (bundled only in the fat jar).</li>
 *   <li>{@link net.ladenthin.llama.server.OpenAiCompatServer} — a dependency-free server built only
 *       on the JDK's {@code com.sun.net.httpserver.HttpServer}. It serves
 *       {@code POST /v1/chat/completions} (streaming via Server-Sent Events and non-streaming) and
 *       {@code GET /v1/models}. Streaming comes straight from the native OpenAI chunk formatter (see
 *       {@link net.ladenthin.llama.LlamaModel#streamChatCompletion(net.ladenthin.llama.parameters.InferenceParameters, java.util.function.Consumer)}),
 *       so streamed {@code delta.tool_calls} are preserved for agent-mode tool use.</li>
 * </ul>
 *
 * <p>JSpecify {@code @NullMarked} is applied module-wide (see {@code module-info.java}) and applies
 * to this package transitively.</p>
 */
package net.ladenthin.llama.server;
