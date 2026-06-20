// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.util.Objects;
import lombok.ToString;

/**
 * Scaffold for the <em>native</em> HTTP server bridge — the planned counterpart to
 * {@link OpenAiCompatServer}.
 *
 * <p>{@link OpenAiCompatServer} implements the HTTP transport in Java (on the JDK's
 * {@code com.sun.net.httpserver}) and drives the native llama.cpp server <em>core</em> over JNI. This
 * class is instead the entry point for the upstream <em>native</em> HTTP transport that is already
 * compiled into {@code libjllama} (llama.cpp's {@code server-http.cpp} plus its {@code cpp-httplib}
 * backend). That native transport is the only component able to serve the embedded llama.cpp
 * <strong>WebUI</strong> (the {@code ui.cpp}/{@code ui.h} asset table compiled in behind
 * {@code LLAMA_UI_HAS_ASSETS}).</p>
 *
 * <p><strong>Status: scaffold only.</strong> The route registration that upstream performs in
 * {@code server.cpp} (deliberately excluded from this build) is not yet wired to a JNI entry point, so
 * {@link #start()} throws {@link UnsupportedOperationException} for now. This class only fixes the
 * package structure and the public API shape; the native {@code startServer}/{@code stopServer}
 * methods, their C++ implementation, the server lifecycle/threading and WebUI serving are a separate,
 * detailed step (see {@code CLAUDE.md}, "WebUI (llama.cpp Svelte UI) embedding").</p>
 *
 * <p>It is {@link AutoCloseable} so that, once implemented, callers can drive it with
 * try-with-resources exactly like {@link OpenAiCompatServer}.</p>
 */
@ToString
public final class NativeServer implements AutoCloseable {

    /** Message thrown by {@link #start()} until the native route-wiring lands. */
    static final String NOT_WIRED_MESSAGE =
            "NativeServer is a scaffold: the upstream native HTTP routes (server-http.cpp) are "
                    + "not yet wired to JNI. Use OpenAiCompatServer for now; the native server and "
                    + "embedded WebUI are a planned step.";

    /** Immutable server configuration (bind host, port, ...) shared with {@link OpenAiCompatServer}. */
    private final OpenAiServerConfig config;

    /**
     * Creates a native-server bridge for the given configuration.
     *
     * <p>Construction performs no native work and binds no socket; it only captures the configuration.
     * Call {@link #start()} to launch the server (not implemented yet).</p>
     *
     * @param config the server configuration (host, port, ...); must not be {@code null}
     */
    public NativeServer(OpenAiServerConfig config) {
        this.config = Objects.requireNonNull(config, "config");
    }

    /**
     * Starts the native HTTP server and begins serving the embedded WebUI.
     *
     * <p><strong>Not implemented yet</strong> — this is a scaffold. The native route registration and
     * its JNI binding are a planned step, so this method always throws until then.</p>
     *
     * @return this server instance (for fluent / try-with-resources use), once implemented
     * @throws UnsupportedOperationException always, until the native routes are wired to JNI
     */
    // Scaffold: start() intentionally always throws for now, but must stay callable (not @DoNotCall)
    // so the real implementation and its callers/tests keep the same signature.
    @SuppressWarnings("DoNotCallSuggester")
    public NativeServer start() {
        throw new UnsupportedOperationException(NOT_WIRED_MESSAGE);
    }

    /**
     * Reports whether the native server is currently running.
     *
     * @return {@code false} — the scaffold never starts a server yet
     */
    public boolean isRunning() {
        return false;
    }

    /**
     * Returns the host the server is configured to bind to.
     *
     * @return the configured bind host
     */
    public String getHost() {
        return config.getHost();
    }

    /**
     * Returns the port the server is configured to bind to.
     *
     * @return the configured port
     */
    public int getPort() {
        return config.getPort();
    }

    /**
     * Stops the native server if it is running.
     *
     * <p>No-op in the scaffold (nothing is ever started), so it is always safe to call, including from
     * try-with-resources. Real lifecycle teardown is part of the planned native-server implementation.</p>
     */
    @Override
    public void close() {
        // Nothing is started yet, so there is nothing to release.
    }
}
