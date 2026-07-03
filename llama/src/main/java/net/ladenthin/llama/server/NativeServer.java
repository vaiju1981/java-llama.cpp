// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import lombok.ToString;
import net.ladenthin.llama.loader.LlamaLoader;

/**
 * Runs the <em>full</em> upstream llama.cpp HTTP server — including its embedded
 * <strong>WebUI</strong> — inside {@code libjllama}, driven over JNI, with no separate
 * {@code llama-server} executable. It is the second of two server modes, the native counterpart to
 * the Java-transport {@link OpenAiCompatServer}.
 *
 * <p>The constructor takes the raw llama-server command-line arguments and forwards them verbatim
 * to the native entry point ({@code llama_server}), so <em>every</em> llama-server flag is supported
 * ({@code -m}, {@code -c}, {@code -b}, {@code -ub}, {@code -ngl}, {@code -t}, {@code -tb},
 * {@code -ctk}, {@code -ctv}, {@code --jinja}, {@code --chat-template-kwargs}, {@code --host},
 * {@code --port}, {@code --ui}/{@code --no-ui}, …). Unlike {@link OpenAiCompatServer}, no per-flag
 * Java mapping is involved.</p>
 *
 * <p><strong>Independent lifecycle.</strong> {@code NativeServer} loads its <em>own</em> model from
 * the forwarded arguments — exactly like running {@code llama-server.exe} — and is unrelated to any
 * {@code net.ladenthin.llama.LlamaModel} you may also have open. Reusing an already-loaded
 * {@code LlamaModel}'s context instead of loading a second copy is a possible future enhancement
 * (see {@code TODO.md}). While the native server runs it owns the process-wide llama backend and
 * routes llama.cpp logging to stderr/file (llama-server's own logging), not the JNI log callback.</p>
 *
 * <p><strong>Single instance per process.</strong> The upstream server keeps its shutdown state in
 * file-scope globals, so only one {@code NativeServer} may run at a time; {@link #start()} throws if
 * another instance is already running.</p>
 *
 * <p>Typical use:</p>
 * <pre>{@code
 * try (NativeServer server = new NativeServer(
 *         "-m", "models/model.gguf", "--host", "127.0.0.1", "--port", "8080", "-c", "65536").start()) {
 *     // Server (and WebUI at http://127.0.0.1:8080/) runs on a native worker thread.
 *     // Readiness: poll GET /health until it returns {"status":"ok"}.
 *     Thread.currentThread().join();
 * }
 * }</pre>
 *
 * <p><strong>Platform note.</strong> The native methods are compiled into {@code libjllama} on all
 * platforms except Android (the upstream server pulls in {@code posix_spawn_*}, unavailable there);
 * on Android use {@link OpenAiCompatServer}. No SSL: the embedded server is plain HTTP — bind
 * localhost or front it with a TLS proxy.</p>
 */
@ToString
public final class NativeServer implements AutoCloseable {

    /** Guards the process-wide single-instance invariant (upstream uses file-scope globals). */
    private static final AtomicBoolean RUNNING = new AtomicBoolean(false);

    /** Default bind host reported by {@link #getHost()} when {@code --host} is not passed. */
    private static final String DEFAULT_HOST = "127.0.0.1";

    /** Default port reported by {@link #getPort()} when no port flag is passed. */
    private static final int DEFAULT_PORT = 8080;

    /** The llama-server argument vector, forwarded verbatim to the native entry point. */
    private final String[] args;

    /** Native handle (pointer) while running, or {@code 0} when not started / stopped. */
    private volatile long handle;

    /**
     * Creates a native-server bridge for the given llama-server arguments.
     *
     * <p>Construction performs no native work and binds no socket; it only captures the arguments.
     * Call {@link #start()} to launch the server.</p>
     *
     * @param args the llama-server command-line arguments (e.g. {@code "-m", "model.gguf",
     *             "--port", "8080"}); must not be {@code null} and must not contain {@code null}
     *             elements
     */
    public NativeServer(String... args) {
        Objects.requireNonNull(args, "args");
        for (final String arg : args) {
            Objects.requireNonNull(arg, "args element");
        }
        this.args = args.clone();
    }

    /**
     * Starts the native HTTP server (and its embedded WebUI) on a background thread and returns
     * immediately. The server binds and begins serving {@code GET /health} before the model finishes
     * loading; poll {@code /health} for readiness.
     *
     * @return this server instance (for fluent / try-with-resources use)
     * @throws IllegalStateException if this instance was already started, or another
     *                               {@code NativeServer} is already running in this process
     */
    public NativeServer start() {
        if (handle != 0) {
            throw new IllegalStateException("NativeServer already started");
        }
        if (!RUNNING.compareAndSet(false, true)) {
            throw new IllegalStateException(
                    "another NativeServer is already running in this process (only one is supported)");
        }
        try {
            // Load libjllama lazily here (not in a static initializer) so construction, argument
            // parsing and close() stay usable — and unit-testable — without the native library.
            LlamaLoader.initialize();
            handle = startNativeServer(args);
        } catch (final RuntimeException | Error e) {
            RUNNING.set(false);
            throw e;
        }
        return this;
    }

    /**
     * Reports whether the native server worker is currently running.
     *
     * <p>Note: this becomes {@code true} as soon as the worker thread starts, which is before the
     * socket is necessarily accepting connections — use {@code GET /health} to detect readiness.</p>
     *
     * @return {@code true} if the server has been started and its worker has not yet exited
     */
    public boolean isRunning() {
        final long h = handle;
        return h != 0 && isRunningNative(h);
    }

    /**
     * Returns the bind host parsed from the arguments ({@code --host}), or {@code 127.0.0.1} when
     * absent. Best-effort convenience for logging; the authoritative value is what the native server
     * parsed.
     *
     * @return the configured bind host
     */
    public String getHost() {
        for (int i = 0; i < args.length - 1; i++) {
            if ("--host".equals(args[i])) {
                return args[i + 1];
            }
        }
        return DEFAULT_HOST;
    }

    /**
     * Returns the port parsed from the arguments ({@code --port} / {@code -p}), or {@code 8080} when
     * absent or unparseable. Best-effort convenience for logging.
     *
     * @return the configured port
     */
    public int getPort() {
        for (int i = 0; i < args.length - 1; i++) {
            if ("--port".equals(args[i]) || "-p".equals(args[i])) {
                try {
                    return Integer.parseInt(args[i + 1].trim());
                } catch (final NumberFormatException e) {
                    return DEFAULT_PORT;
                }
            }
        }
        return DEFAULT_PORT;
    }

    /**
     * Stops the native server if it is running and releases the native handle. Blocks until the
     * server has fully shut down. Safe to call more than once and from try-with-resources even if
     * {@link #start()} was never called (no-op then).
     */
    @Override
    public void close() {
        final long h = handle;
        if (h == 0) {
            return;
        }
        handle = 0;
        try {
            stopNativeServer(h);
        } finally {
            RUNNING.set(false);
        }
    }

    /**
     * Fat-jar entry point (the assembly JAR's {@code Main-Class}): starts the full native llama.cpp
     * server — WebUI included — forwarding every argument to it verbatim, and blocks until the
     * server exits or the JVM is asked to shut down (Ctrl-C / SIGTERM), stopping the server cleanly
     * on the way out.
     *
     * <p>This is the default runnable server. The Java-transport {@link OpenAiCompatServer} remains
     * available via its own {@code main} — run it explicitly with
     * {@code java -cp <jar> net.ladenthin.llama.server.OpenAiCompatServer …}.</p>
     *
     * @param args the llama-server command-line arguments, forwarded verbatim (e.g. {@code -m
     *             model.gguf --host 127.0.0.1 --port 8080}); pass {@code --help} for the full
     *             llama-server option list
     * @throws InterruptedException if interrupted while waiting for the server to exit
     */
    public static void main(String[] args) throws InterruptedException {
        final NativeServer server = new NativeServer(args);
        final AtomicBoolean stoppedByHook = new AtomicBoolean(false);
        // Graceful Ctrl-C / SIGTERM: the embedded server installs no signal handlers of its own
        // (see patches/0006), so the JVM-level shutdown hook is what stops it before exit.
        Runtime.getRuntime()
                .addShutdownHook(new Thread(
                        () -> {
                            stoppedByHook.set(true);
                            server.close();
                        },
                        "jllama-native-server-shutdown"));
        server.start();
        // Keep the JVM alive until the native worker exits — on its own (e.g. a fatal startup/model
        // error that llama_server has already logged) or because the shutdown hook stopped it.
        while (server.isRunning()) {
            Thread.sleep(200L);
        }
        if (!stoppedByHook.get()) {
            server.close();
        }
    }

    /**
     * Starts the native server on a worker thread and returns an opaque handle. The argv is
     * forwarded verbatim (with a synthetic {@code argv[0]}).
     */
    private static native long startNativeServer(String[] args);

    /** Signals shutdown, joins the worker thread, and frees the handle. */
    private static native void stopNativeServer(long handle);

    /** Whether the worker thread for the given handle is still running. */
    private static native boolean isRunningNative(long handle);
}
