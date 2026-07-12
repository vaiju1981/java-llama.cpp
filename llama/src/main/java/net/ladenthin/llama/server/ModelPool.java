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
        this.baseUrl = baseUrl;
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
