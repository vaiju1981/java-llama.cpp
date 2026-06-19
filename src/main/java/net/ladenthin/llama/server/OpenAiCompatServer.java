// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import net.ladenthin.llama.LlamaModel;
import org.jspecify.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An OpenAI-compatible HTTP endpoint over a loaded {@link LlamaModel}, built only on the JDK's
 * {@code com.sun.net.httpserver.HttpServer} (no new runtime dependency). It is both embeddable and the
 * {@code Main-Class} of the {@code -jar-with-dependencies} assembly.
 *
 * <p>Routes:
 * <ul>
 *   <li>{@code POST /v1/chat/completions} — streaming (Server-Sent Events) and non-streaming chat
 *       completions, forwarded faithfully (messages/tools verbatim; streamed {@code delta.tool_calls}
 *       preserved).</li>
 *   <li>{@code POST /v1/completions} — non-streaming text completion.</li>
 *   <li>{@code POST /v1/embeddings} — embeddings (requires the model to be loaded in embedding
 *       mode).</li>
 *   <li>{@code GET /v1/models} — advertises the single configured model.</li>
 *   <li>{@code GET /health} — liveness probe returning {@code {"status":"ok"}} (no authentication).</li>
 * </ul>
 *
 * <p>During streaming, the server emits SSE comment heartbeats on a timer so a long prompt prefill on
 * CPU does not trip a client's stream-inactivity timeout before the first token. It binds to loopback by
 * default and can require a bearer API key. The endpoint is a pass-through: tools are provided and
 * executed by the client, not here.
 *
 * <p>Typical use:
 * <pre>{@code
 * try (LlamaModel model = new LlamaModel(new ModelParameters().setModel("models/model.gguf"));
 *      OpenAiCompatServer server = new OpenAiCompatServer(
 *              model, OpenAiServerConfig.builder().port(8080).modelId("local").build()).start()) {
 *     Thread.currentThread().join();
 * }
 * }</pre>
 */
public final class OpenAiCompatServer implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(OpenAiCompatServer.class);
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /** The chat-completions route. */
    public static final String PATH_CHAT_COMPLETIONS = "/v1/chat/completions";

    /** The text-completions route. */
    public static final String PATH_COMPLETIONS = "/v1/completions";

    /** The embeddings route. */
    public static final String PATH_EMBEDDINGS = "/v1/embeddings";

    /** The model-list route. */
    public static final String PATH_MODELS = "/v1/models";

    /** The liveness-probe route. */
    public static final String PATH_HEALTH = "/health";

    private static final int HTTP_OK = 200;
    private static final int HTTP_BAD_REQUEST = 400;
    private static final int HTTP_UNAUTHORIZED = 401;
    private static final int HTTP_NOT_FOUND = 404;
    private static final int HTTP_METHOD_NOT_ALLOWED = 405;
    private static final int HTTP_SERVER_ERROR = 500;

    private static final String CONTENT_TYPE_JSON = "application/json; charset=utf-8";
    private static final String CONTENT_TYPE_SSE = "text/event-stream; charset=utf-8";
    private static final String BEARER_PREFIX = "Bearer ";
    private static final String ERROR_TYPE_REQUEST = "invalid_request_error";
    private static final String ERROR_TYPE_SERVER = "server_error";
    private static final String HEALTH_BODY = "{\"status\":\"ok\"}";

    private final OpenAiServerConfig config;
    private final OpenAiBackend backend;
    private final HttpServer http;
    private final ExecutorService requestExecutor;
    private final ScheduledExecutorService heartbeatExecutor;

    /**
     * Create a server backed by a loaded model.
     *
     * @param model the model to serve completions from (owned by the caller; not closed by the server)
     * @param config the server configuration
     * @throws IOException if the listening socket cannot be bound
     */
    public OpenAiCompatServer(LlamaModel model, OpenAiServerConfig config) throws IOException {
        this(new LlamaModelBackend(model, new OpenAiRequestMapper()), config);
    }

    /**
     * Create a server backed by an arbitrary {@link OpenAiBackend}. Used by tests to drive the full HTTP
     * surface without a native library or model.
     *
     * @param backend the inference engine seam
     * @param config the server configuration
     * @throws IOException if the listening socket cannot be bound
     */
    OpenAiCompatServer(OpenAiBackend backend, OpenAiServerConfig config) throws IOException {
        this.config = config;
        this.backend = backend;
        this.requestExecutor = Executors.newCachedThreadPool(namedFactory("jllama-openai-http"));
        this.heartbeatExecutor = Executors.newScheduledThreadPool(1, namedFactory("jllama-openai-hb"));
        this.http = HttpServer.create(new InetSocketAddress(config.getHost(), config.getPort()), 0);
        http.createContext("/", this::handleNotFound);
        http.createContext(PATH_HEALTH, this::handleHealth);
        http.createContext(PATH_MODELS, this::handleModels);
        http.createContext(PATH_CHAT_COMPLETIONS, this::handleChatCompletions);
        http.createContext(PATH_COMPLETIONS, this::handleCompletions);
        http.createContext(PATH_EMBEDDINGS, this::handleEmbeddings);
        http.setExecutor(requestExecutor);
    }

    /**
     * Start accepting connections.
     *
     * @return this server, for chaining
     */
    public OpenAiCompatServer start() {
        http.start();
        LOG.info("OpenAI-compatible server listening on http://{}:{}", config.getHost(), getPort());
        return this;
    }

    /**
     * The actual bound port (useful when configured with port {@code 0} for an ephemeral port).
     *
     * @return the port the server is listening on
     */
    public int getPort() {
        return http.getAddress().getPort();
    }

    /** Stop the server and release its thread pools. The backing model is not closed. */
    @Override
    public void close() {
        http.stop(0);
        requestExecutor.shutdownNow();
        heartbeatExecutor.shutdownNow();
    }

    // ----- handlers -----

    private void handleChatCompletions(HttpExchange exchange) throws IOException {
        try {
            JsonNode request = requirePostJson(exchange);
            if (request == null) {
                return;
            }
            JsonNode messages = request.path("messages");
            if (!messages.isArray() || messages.size() == 0) {
                sendError(exchange, HTTP_BAD_REQUEST, ERROR_TYPE_REQUEST, "'messages' must be a non-empty array");
                return;
            }
            if (request.path("stream").asBoolean(false)) {
                streamChat(exchange, request);
            } else {
                completeNonStreaming(exchange, request, backend::complete);
            }
        } finally {
            exchange.close();
        }
    }

    private void handleCompletions(HttpExchange exchange) throws IOException {
        try {
            JsonNode request = requirePostJson(exchange);
            if (request != null) {
                completeNonStreaming(exchange, request, backend::completions);
            }
        } finally {
            exchange.close();
        }
    }

    private void handleEmbeddings(HttpExchange exchange) throws IOException {
        try {
            JsonNode request = requirePostJson(exchange);
            if (request != null) {
                completeNonStreaming(exchange, request, backend::embeddings);
            }
        } finally {
            exchange.close();
        }
    }

    /**
     * Run a non-streaming request through {@code producer} and write its JSON body, translating an
     * {@link IllegalArgumentException} to {@code 400} and any other failure to {@code 500}.
     */
    private void completeNonStreaming(HttpExchange exchange, JsonNode request, BodyProducer producer)
            throws IOException {
        final String body;
        try {
            body = producer.produce(request);
        } catch (IllegalArgumentException e) {
            sendError(exchange, HTTP_BAD_REQUEST, ERROR_TYPE_REQUEST, message(e));
            return;
        } catch (IOException | RuntimeException e) {
            LOG.warn("request failed", e);
            sendError(exchange, HTTP_SERVER_ERROR, ERROR_TYPE_SERVER, message(e));
            return;
        }
        sendJson(exchange, HTTP_OK, body);
    }

    private void streamChat(HttpExchange exchange, JsonNode request) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", CONTENT_TYPE_SSE);
        exchange.getResponseHeaders().set("Cache-Control", "no-cache");
        exchange.sendResponseHeaders(HTTP_OK, 0);
        final OutputStream os = exchange.getResponseBody();
        final Object writeLock = new Object();
        final ScheduledFuture<?> heartbeat = heartbeatExecutor.scheduleAtFixedRate(
                () -> writeQuietly(os, writeLock, OpenAiSseFormatter.heartbeat()),
                config.getHeartbeatMillis(),
                config.getHeartbeatMillis(),
                TimeUnit.MILLISECONDS);
        try {
            backend.stream(request, chunkJson -> writeStrict(os, writeLock, OpenAiSseFormatter.sseData(chunkJson)));
            writeStrict(os, writeLock, OpenAiSseFormatter.sseDone());
        } catch (IllegalArgumentException e) {
            writeQuietly(
                    os,
                    writeLock,
                    OpenAiSseFormatter.sseData(OpenAiSseFormatter.errorJson(message(e), ERROR_TYPE_REQUEST, null)));
        } catch (IOException e) {
            LOG.debug("client disconnected during stream", e);
        } catch (RuntimeException e) {
            LOG.warn("streaming chat completion failed", e);
            writeQuietly(
                    os,
                    writeLock,
                    OpenAiSseFormatter.sseData(OpenAiSseFormatter.errorJson(message(e), ERROR_TYPE_SERVER, null)));
        } finally {
            heartbeat.cancel(false);
            closeQuietly(os, writeLock);
        }
    }

    private void handleModels(HttpExchange exchange) throws IOException {
        try {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendError(exchange, HTTP_METHOD_NOT_ALLOWED, ERROR_TYPE_REQUEST, "Only GET is supported");
                return;
            }
            if (!authorized(exchange)) {
                sendError(exchange, HTTP_UNAUTHORIZED, ERROR_TYPE_REQUEST, "Missing or invalid API key");
                return;
            }
            sendJson(exchange, HTTP_OK, OpenAiSseFormatter.modelsJson(config.getModelId()));
        } finally {
            exchange.close();
        }
    }

    private void handleHealth(HttpExchange exchange) throws IOException {
        try {
            // Liveness probe: deliberately unauthenticated so orchestrators can poll it without a key.
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendError(exchange, HTTP_METHOD_NOT_ALLOWED, ERROR_TYPE_REQUEST, "Only GET is supported");
                return;
            }
            sendJson(exchange, HTTP_OK, HEALTH_BODY);
        } finally {
            exchange.close();
        }
    }

    private void handleNotFound(HttpExchange exchange) throws IOException {
        try {
            sendError(exchange, HTTP_NOT_FOUND, ERROR_TYPE_REQUEST, "Not found: " + exchange.getRequestURI());
        } finally {
            exchange.close();
        }
    }

    // ----- helpers -----

    /**
     * Shared preamble for the {@code POST} JSON routes: enforce the method, authentication and a JSON
     * object body, sending the matching error and returning {@code null} when any precondition fails.
     */
    private @Nullable JsonNode requirePostJson(HttpExchange exchange) throws IOException {
        if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
            sendError(exchange, HTTP_METHOD_NOT_ALLOWED, ERROR_TYPE_REQUEST, "Only POST is supported");
            return null;
        }
        if (!authorized(exchange)) {
            sendError(exchange, HTTP_UNAUTHORIZED, ERROR_TYPE_REQUEST, "Missing or invalid API key");
            return null;
        }
        JsonNode request = readBody(exchange);
        if (request == null || !request.isObject()) {
            sendError(exchange, HTTP_BAD_REQUEST, ERROR_TYPE_REQUEST, "Request body must be a JSON object");
            return null;
        }
        return request;
    }

    private boolean authorized(HttpExchange exchange) {
        if (!config.isAuthenticationEnabled()) {
            return true;
        }
        String expected = config.getApiKey();
        if (expected == null) {
            return true;
        }
        String header = exchange.getRequestHeaders().getFirst("Authorization");
        if (header == null || !header.startsWith(BEARER_PREFIX)) {
            return false;
        }
        return expected.equals(header.substring(BEARER_PREFIX.length()));
    }

    private @Nullable JsonNode readBody(HttpExchange exchange) throws IOException {
        try (InputStream is = exchange.getRequestBody()) {
            return OBJECT_MAPPER.readTree(is);
        } catch (JsonProcessingException e) {
            LOG.debug("malformed request body", e);
            return null;
        }
    }

    private void sendJson(HttpExchange exchange, int status, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", CONTENT_TYPE_JSON);
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    private void sendError(HttpExchange exchange, int status, String type, String message) throws IOException {
        sendJson(exchange, status, OpenAiSseFormatter.errorJson(message, type, null));
    }

    /** Write under the response lock, propagating failures so a streaming generation can be cancelled. */
    private void writeStrict(OutputStream os, Object writeLock, String text) throws IOException {
        synchronized (writeLock) {
            os.write(text.getBytes(StandardCharsets.UTF_8));
            os.flush();
        }
    }

    /** Write under the response lock, swallowing failures (used for heartbeats and best-effort events). */
    private void writeQuietly(OutputStream os, Object writeLock, String text) {
        synchronized (writeLock) {
            try {
                os.write(text.getBytes(StandardCharsets.UTF_8));
                os.flush();
            } catch (IOException e) {
                LOG.trace("stream write failed (client likely disconnected)", e);
            }
        }
    }

    private void closeQuietly(OutputStream os, Object writeLock) {
        synchronized (writeLock) {
            try {
                os.close();
            } catch (IOException e) {
                LOG.trace("stream close failed", e);
            }
        }
    }

    private static String message(Throwable t) {
        String m = t.getMessage();
        return m != null ? m : t.getClass().getSimpleName();
    }

    private static ThreadFactory namedFactory(String prefix) {
        AtomicInteger counter = new AtomicInteger();
        return runnable -> {
            Thread thread = new Thread(runnable, prefix + "-" + counter.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        };
    }

    /** Produces a non-streaming response body from a parsed request; may fail with {@link IOException}. */
    @FunctionalInterface
    private interface BodyProducer {
        String produce(JsonNode request) throws IOException;
    }

    // ----- standalone launcher -----

    /**
     * Command-line launcher: load a GGUF model and serve it over the OpenAI-compatible endpoint. This is
     * the {@code Main-Class} of the {@code -jar-with-dependencies} assembly.
     *
     * <p>Parsing, validation and the option list live in {@link OpenAiServerCli}; run with
     * {@code --help} for the full usage text. No {@code System.exit} is used (the {@code noSystemExit}
     * architecture rule forbids it): a usage error prints to stderr and returns.
     *
     * @param args command-line options
     * @throws IOException if the listening socket cannot be bound
     */
    public static void main(String[] args) throws IOException {
        if (OpenAiServerCli.isHelpRequested(args)) {
            System.out.println(OpenAiServerCli.usage());
            return;
        }

        final OpenAiServerCli.Options options;
        try {
            options = OpenAiServerCli.parse(args);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            return;
        }

        OpenAiServerConfig config = options.toServerConfig();
        LlamaModel model = new LlamaModel(options.toModelParameters());
        OpenAiCompatServer server = new OpenAiCompatServer(model, config);
        Runtime.getRuntime()
                .addShutdownHook(new Thread(
                        () -> {
                            server.close();
                            model.close();
                        },
                        "jllama-openai-shutdown"));
        server.start();
        printReady(config, server.getPort());
        try {
            Thread.currentThread().join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static void printReady(OpenAiServerConfig config, int port) {
        String url = "http://" + config.getHost() + ":" + port + PATH_CHAT_COMPLETIONS;
        System.out.println();
        System.out.println("OpenAI-compatible endpoint ready: " + url);
        System.out.println("Add this to VS Code's chatLanguageModels.json (Chat: Manage Language Models):");
        System.out.println("[");
        System.out.println("  {");
        System.out.println("    \"name\": \"Local llama.cpp (java-llama.cpp)\",");
        System.out.println("    \"vendor\": \"customendpoint\",");
        System.out.println(
                "    \"apiKey\": \"" + (config.isAuthenticationEnabled() ? "<your key>" : "local-dummy-key") + "\",");
        System.out.println("    \"apiType\": \"chat-completions\",");
        System.out.println("    \"models\": [");
        System.out.println("      {");
        System.out.println("        \"id\": \"" + config.getModelId() + "\",");
        System.out.println("        \"name\": \"" + config.getModelId() + "\",");
        System.out.println("        \"url\": \"" + url + "\",");
        System.out.println("        \"toolCalling\": true,");
        System.out.println("        \"vision\": false,");
        System.out.println("        \"maxInputTokens\": " + config.getMaxInputTokens() + ",");
        System.out.println("        \"maxOutputTokens\": " + config.getMaxOutputTokens());
        System.out.println("      }");
        System.out.println("    ]");
        System.out.println("  }");
        System.out.println("]");
    }
}
