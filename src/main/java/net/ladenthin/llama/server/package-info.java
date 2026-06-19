// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

/**
 * Optional OpenAI-compatible HTTP server over a loaded {@link net.ladenthin.llama.LlamaModel}.
 *
 * <p>{@link net.ladenthin.llama.server.OpenAiCompatServer} is a dependency-free server built only on
 * the JDK's {@code com.sun.net.httpserver.HttpServer} (the supported, exported {@code jdk.httpserver}
 * module — no web-framework dependency). It is both embeddable and the {@code Main-Class} of the
 * {@code -jar-with-dependencies} assembly, so editors and tools that speak the OpenAI protocol (for
 * example a VS&nbsp;Code Copilot "Custom Endpoint") can drive a local GGUF model running in-process
 * through the JNI binding. It is a faithful pass-through that does not implement or execute tools
 * itself.</p>
 *
 * <p>Routes:</p>
 * <ul>
 *   <li>{@code POST /v1/chat/completions} — streaming (Server-Sent Events) and non-streaming. Streaming
 *       comes straight from the native OpenAI chunk formatter (see
 *       {@link net.ladenthin.llama.LlamaModel#streamChatCompletion(net.ladenthin.llama.parameters.InferenceParameters, java.util.function.Consumer)}),
 *       so streamed {@code delta.tool_calls} are preserved for agent-mode tool use.</li>
 *   <li>{@code POST /v1/completions} and {@code POST /v1/embeddings} — non-streaming, forwarding the
 *       request body to the matching {@code LlamaModel.handle*} method.</li>
 *   <li>{@code POST /infill} — non-streaming fill-in-the-middle for local ghost-text autocomplete
 *       clients (llama.vscode, Twinny, Tabby); the model's FIM tokens are applied server-side.</li>
 *   <li>{@code GET /v1/models} — advertises the configured model id.</li>
 *   <li>{@code GET /health} — unauthenticated liveness probe.</li>
 * </ul>
 *
 * <p>Every route is also reachable without the {@code /v1} prefix, answers CORS preflight
 * ({@code OPTIONS}) requests, and stamps {@code Access-Control-Allow-Origin} on responses so
 * browser/webview clients are not blocked.</p>
 *
 * <p>The HTTP surface is decoupled from the model behind {@link net.ladenthin.llama.server.OpenAiBackend}
 * (production implementation {@link net.ladenthin.llama.server.LlamaModelBackend}) so routing,
 * authentication, SSE framing and heartbeats are unit-testable with a fake backend — no socket and no
 * native model. The standalone launcher's command line is parsed by
 * {@link net.ladenthin.llama.server.OpenAiServerCli}.</p>
 *
 * <p>JSpecify {@code @NullMarked} is applied module-wide (see {@code module-info.java}) and applies
 * to this package transitively.</p>
 */
package net.ladenthin.llama.server;
