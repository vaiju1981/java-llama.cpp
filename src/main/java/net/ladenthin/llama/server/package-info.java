// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

/**
 * A minimal, dependency-free OpenAI-compatible HTTP endpoint over a loaded
 * {@link net.ladenthin.llama.LlamaModel}.
 *
 * <p>{@link net.ladenthin.llama.server.OpenAiCompatServer} serves
 * {@code POST /v1/chat/completions} (streaming via Server-Sent Events and non-streaming) and
 * {@code GET /v1/models} using only the JDK's built-in {@code com.sun.net.httpserver.HttpServer}
 * — no web framework and no new runtime dependency. It exists so editors and tools that speak the
 * OpenAI Chat Completions protocol (for example a VS&nbsp;Code Copilot "Custom Endpoint") can drive a
 * local GGUF model running in-process through the JNI binding.
 *
 * <p>Streaming responses come straight from the native OpenAI chunk formatter
 * (see {@link net.ladenthin.llama.LlamaModel#streamChatCompletion}), so streamed
 * {@code delta.tool_calls} are preserved for agent-mode tool use. The endpoint is a faithful
 * pass-through: it does not implement or execute tools itself.
 *
 * <p>JSpecify {@code @NullMarked} is declared at module level in {@code module-info.java} and applies
 * to this package transitively.
 */
package net.ladenthin.llama.server;
