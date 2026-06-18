// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.util.function.Function;
import lombok.ToString;
import org.jspecify.annotations.Nullable;

/**
 * Maps an HTTP method + path + body to an {@link OaiResponse} by dispatching to an
 * {@link OaiBackend}. This is the testable core of the server: it is independent of NanoHTTPD and
 * of {@link net.ladenthin.llama.LlamaModel}, so it can be exercised with a fake backend and plain
 * strings (no socket, no native library, no GGUF model).
 *
 * <p>Supported routes:</p>
 * <ul>
 *   <li>{@code POST /v1/chat/completions} &rarr; {@link OaiBackend#chatCompletions(String)}</li>
 *   <li>{@code POST /v1/completions} &rarr; {@link OaiBackend#completions(String)}</li>
 *   <li>{@code POST /v1/embeddings} &rarr; {@link OaiBackend#embeddings(String)}</li>
 *   <li>{@code GET /v1/models} &rarr; {@link OaiBackend#listModels()}</li>
 *   <li>{@code GET /health} and {@code GET /} &rarr; a static {@code {"status":"ok"}}</li>
 * </ul>
 *
 * <p>Unknown paths yield {@code 404}; a known path with the wrong method yields {@code 405}; an
 * empty body on a {@code POST} route yields {@code 400}; any {@link RuntimeException} thrown by the
 * backend (e.g. inference failure) is converted to {@code 500}. Error bodies use the OpenAI error
 * envelope {@code {"error":{"message":...,"type":...}}}.</p>
 */
@ToString
public final class OaiRouter {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private static final String METHOD_GET = "GET";
    private static final String METHOD_POST = "POST";

    private static final String HEALTH_BODY = "{\"status\":\"ok\"}";

    private final OaiBackend backend;

    /**
     * Create a router over a backend.
     *
     * @param backend the inference backend requests are dispatched to
     */
    public OaiRouter(OaiBackend backend) {
        this.backend = backend;
    }

    /**
     * Route a single request.
     *
     * @param method  the HTTP method (e.g. {@code "GET"}, {@code "POST"})
     * @param rawPath the request path, optionally including a {@code ?query} suffix
     * @param body    the request body, or {@code null} when there is none (e.g. for {@code GET})
     * @return the status code and JSON body to return to the client
     */
    public OaiResponse route(String method, String rawPath, @Nullable String body) {
        final String path = stripQuery(rawPath);
        try {
            switch (path) {
                case "/v1/chat/completions":
                    return post(method, body, backend::chatCompletions);
                case "/v1/completions":
                    return post(method, body, backend::completions);
                case "/v1/embeddings":
                    return post(method, body, backend::embeddings);
                case "/v1/models":
                    return get(method, backend::listModels);
                case "/health":
                case "/":
                    return get(method, () -> HEALTH_BODY);
                default:
                    return error(404, "not_found", "Unknown endpoint: " + path);
            }
        } catch (RuntimeException e) {
            return error(500, "internal_error", describe(e));
        }
    }

    private OaiResponse post(String method, @Nullable String body, Function<String, String> handler) {
        if (!METHOD_POST.equals(method)) {
            return methodNotAllowed(method);
        }
        if (body == null || body.trim().isEmpty()) {
            return error(400, "invalid_request_error", "Request body is required");
        }
        return new OaiResponse(200, handler.apply(body));
    }

    private OaiResponse get(String method, java.util.function.Supplier<String> handler) {
        if (!METHOD_GET.equals(method)) {
            return methodNotAllowed(method);
        }
        return new OaiResponse(200, handler.get());
    }

    private OaiResponse methodNotAllowed(String method) {
        return error(405, "method_not_allowed", "Method not allowed: " + method);
    }

    private static String stripQuery(String rawPath) {
        final int q = rawPath.indexOf('?');
        return q >= 0 ? rawPath.substring(0, q) : rawPath;
    }

    private static String describe(RuntimeException e) {
        final String message = e.getMessage();
        return message != null ? message : e.getClass().getSimpleName();
    }

    private static OaiResponse error(int status, String type, String message) {
        final ObjectNode root = OBJECT_MAPPER.createObjectNode();
        final ObjectNode err = root.putObject("error");
        err.put("message", message);
        err.put("type", type);
        String json;
        try {
            json = OBJECT_MAPPER.writeValueAsString(root);
        } catch (JsonProcessingException e) {
            json = "{\"error\":{\"message\":\"serialization failed\",\"type\":\"internal_error\"}}";
        }
        return new OaiResponse(status, json);
    }
}
