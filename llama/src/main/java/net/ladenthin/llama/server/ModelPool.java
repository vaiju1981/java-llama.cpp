// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.ServerMetrics;
import org.jspecify.annotations.Nullable;

/**
 * Java-side admin facade over a running {@link NativeServer} (router or single-model) for the M3
 * multi-model management milestone.
 *
 * <p>Every method delegates to the upstream llama.cpp HTTP endpoints the native server already
 * exposes — {@code GET/POST /models}, {@code GET /health}, {@code GET /metrics} — so there are no
 * new native calls (consistent with the minimize-JNI rule). {@link #listModels()} reads the model
 * aliases; {@link #loadModel(String, ModelParameters)} / {@link #unloadModel(String)} add and remove
 * models from the pool; {@link #getModelHealth(String)} and {@link #getModelMetrics(String)} read the
 * server's health and metrics.
 *
 * <p>In router mode the upstream server aggregates per-worker state, so {@code getModelHealth} /
 * {@code getModelMetrics} currently reflect the server as a whole (the router's {@code /health} and
 * {@code /metrics}); per-worker breakdown keyed by alias is a follow-up once the router embeds worker
 * addresses in its {@code /models} response.
 */
public final class ModelPool {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private final String baseUrl;
    private final @Nullable ModelRegistry registry;
    private final boolean offline;
    private final @Nullable ModelPuller puller;

    /**
     * Build a pool facade over an already-started {@link NativeServer}, deriving the base URL from the
     * server's parsed bind host and port.
     *
     * @param server the running native server to administer
     */
    public ModelPool(NativeServer server) {
        this("http://" + server.getHost() + ":" + server.getPort());
    }

    /** Package-private constructor for tests that point the facade at a stub HTTP server. */
    ModelPool(String baseUrl) {
        this(baseUrl, null, false, null, ModelPuller.DEFAULT_MODELS_DIR);
    }

    /**
     * Build a pool facade that resolves model aliases against a {@link ModelRegistry} and lazily pulls
     * a model on first load when its local file is absent (the Ollama-style "first request pulls the
     * model" behaviour). A {@code null} registry disables resolution and lazy pull; callers then use
     * {@link #loadModel(String, ModelParameters)} with an explicit path.
     *
     * @param baseUrl the native server's base URL (e.g. {@code http://127.0.0.1:8080})
     * @param registry the registry to resolve aliases from, or {@code null} to disable
     * @param offline when {@code true}, a missing local model is refused instead of being pulled
     */
    public ModelPool(String baseUrl, @Nullable ModelRegistry registry, boolean offline) {
        this(baseUrl, registry, offline, new HttpModelDownloader(), ModelPuller.DEFAULT_MODELS_DIR);
    }

    /**
     * Full constructor (used by tests): supply a custom downloader and model store directory.
     *
     * @param baseUrl the native server's base URL
     * @param registry the registry to resolve aliases from, or {@code null} to disable
     * @param offline when {@code true}, a missing local model is refused instead of being pulled
     * @param downloader the transport used for lazy pulls
     * @param modelsDir directory pulled models are written into
     */
    public ModelPool(
            String baseUrl,
            @Nullable ModelRegistry registry,
            boolean offline,
            @Nullable ModelDownloader downloader,
            Path modelsDir) {
        this.baseUrl = baseUrl;
        this.registry = registry;
        this.offline = offline;
        this.puller = (registry != null && downloader != null)
                ? new ModelPuller(registry, new ModelNameResolver(), downloader, modelsDir, offline)
                : null;
    }

    /**
     * List the model aliases currently registered with the server.
     *
     * @return the model ids/aliases advertised by {@code GET /models}
     * @throws IOException on transport or non-2xx failure
     */
    public List<String> listModels() throws IOException {
        JsonNode node = OBJECT_MAPPER.readTree(get("/models"));
        List<String> ids = new ArrayList<>();
        JsonNode data = node.path("data");
        if (data.isArray()) {
            for (JsonNode model : data) {
                ids.add(model.path("id").asText());
            }
        }
        return ids;
    }

    /**
     * Load a model into the pool.
     *
     * @param alias the alias to register the model under
     * @param params the model parameters (at least the GGUF path via {@link ModelParameters#getModel()})
     * @throws IOException on transport or non-2xx failure
     */
    public void loadModel(String alias, ModelParameters params) throws IOException {
        @Nullable String modelPath = params.getModel();
        StringBuilder body = new StringBuilder();
        body.append("{\"alias\":").append(quote(alias));
        if (modelPath != null) {
            body.append(",\"path\":").append(quote(modelPath));
        }
        body.append('}');
        post("/models", body.toString());
    }

    /**
     * Resolve an alias via the registry and load that model into the pool. When the resolved entry has
     * no local file on disk and the pool is not {@link #isOffline() offline}, the model is pulled first
     * (lazily) and then loaded — the Ollama-style "first request pulls the model" behaviour.
     *
     * @param alias the registry alias to resolve and load
     * @throws IOException on transport, resolution, pull, or non-2xx failure
     * @throws IllegalArgumentException if the alias is absent from the registry
     * @throws IllegalStateException if the pool has no registry, is offline with a missing local model,
     *     or the entry has neither a local path nor a source URL
     */
    public void loadModel(String alias) throws IOException {
        if (registry == null) {
            throw new IllegalStateException("ModelPool has no registry; call loadModel(alias, params) instead");
        }
        ModelRegistryEntry entry = registry.get(alias);
        if (entry == null) {
            throw new IllegalArgumentException("No registry entry for alias: " + alias);
        }
        Path local = localPathOf(entry);
        if (local == null || !Files.exists(local)) {
            if (offline) {
                throw new IllegalStateException("Offline mode: local model missing for '" + alias + "'");
            }
            if (puller == null) {
                throw new IllegalStateException("No puller configured to lazily fetch '" + alias + "'");
            }
            String sourceUrl = entry.getSourceUrl();
            String spec = sourceUrl != null ? sourceUrl : alias;
            ModelRegistryEntry pulled = puller.pull(spec);
            local = localPathOf(pulled);
        }
        if (local == null) {
            throw new IllegalStateException(
                    "Registry entry '" + alias + "' has neither a local path nor a source URL");
        }
        loadModel(alias, new ModelParameters().setModel(local.toString()));
    }

    /**
     * Returns whether lazy pulls are refused.
     *
     * @return whether lazy pulls are refused (offline mode)
     */
    public boolean isOffline() {
        return offline;
    }

    private static @Nullable Path localPathOf(ModelRegistryEntry entry) {
        String localPath = entry.getLocalPath();
        return localPath != null ? Paths.get(localPath) : null;
    }

    /**
     * Unload a model from the pool.
     *
     * @param alias the alias of the model to remove
     * @throws IOException on transport or non-2xx failure
     */
    public void unloadModel(String alias) throws IOException {
        delete("/models/" + alias);
    }

    /**
     * Probe the server's health and measure round-trip latency.
     *
     * @param alias the model alias (currently used for labelling; the probe hits the server's
     *     {@code /health})
     * @return the health probe result
     * @throws IOException on transport or non-2xx failure
     */
    public ModelHealth getModelHealth(String alias) throws IOException {
        long start = System.nanoTime();
        String raw = get("/health");
        long latencyMillis = (System.nanoTime() - start) / 1_000_000L;
        boolean healthy = raw.contains("\"status\"") && !raw.contains("\"status\":\"error\"");
        return new ModelHealth(alias, healthy, latencyMillis, raw);
    }

    /**
     * Read the server's utilization/observability metrics.
     *
     * @param alias the model alias (currently used for labelling; the probe hits the server's
     *     {@code /metrics})
     * @return the parsed metrics
     * @throws IOException on transport or non-2xx failure
     */
    public ServerMetrics getModelMetrics(String alias) throws IOException {
        JsonNode node = OBJECT_MAPPER.readTree(get("/metrics"));
        return new ServerMetrics(node);
    }

    private String get(String path) throws IOException {
        HttpURLConnection conn = open(path, "GET");
        return readBody(conn);
    }

    private String post(String path, String body) throws IOException {
        HttpURLConnection conn = open(path, "POST");
        conn.setDoOutput(true);
        conn.setRequestProperty("Content-Type", "application/json");
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        try (OutputStream os = conn.getOutputStream()) {
            os.write(bytes);
        }
        return readBody(conn);
    }

    private String delete(String path) throws IOException {
        HttpURLConnection conn = open(path, "DELETE");
        return readBody(conn);
    }

    private HttpURLConnection open(String path, String method) throws IOException {
        HttpURLConnection conn = (HttpURLConnection) new URL(baseUrl + path).openConnection();
        conn.setRequestMethod(method);
        conn.setConnectTimeout(5_000);
        conn.setReadTimeout(30_000);
        return conn;
    }

    private static String readBody(HttpURLConnection conn) throws IOException {
        int code = conn.getResponseCode();
        InputStream is = code < 400 ? conn.getInputStream() : conn.getErrorStream();
        String body = is == null ? "" : readAll(is);
        if (code < 200 || code >= 300) {
            throw new IOException("ModelPool request failed: HTTP " + code + " " + body);
        }
        return body;
    }

    private static String readAll(InputStream is) throws IOException {
        StringBuilder out = new StringBuilder();
        byte[] buffer = new byte[1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
            out.append(new String(buffer, 0, read, StandardCharsets.UTF_8));
        }
        return out.toString();
    }

    private static String quote(String value) {
        return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }
}
