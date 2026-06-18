// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import fi.iki.elonen.NanoHTTPD;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import lombok.ToString;
import org.jspecify.annotations.Nullable;

/**
 * Thin NanoHTTPD adapter: reads the method, path and (for body-bearing methods) the raw request
 * body from each session, hands them to an {@link OaiRouter}, and converts the resulting
 * {@link OaiResponse} into a fixed-length {@code application/json} NanoHTTPD response.
 *
 * <p>All request-shaping decisions live in {@link OaiRouter}; this class only bridges NanoHTTPD's
 * session API to that router so the routing logic stays unit-testable without a socket.</p>
 */
@ToString
public final class OaiHttpServer extends NanoHTTPD {

    private static final String MIME_JSON = "application/json";

    private static final String MALFORMED_BODY_JSON =
            "{\"error\":{\"message\":\"Malformed request body\",\"type\":\"invalid_request_error\"}}";

    private final OaiRouter router;

    /**
     * Create (but do not start) the server.
     *
     * @param host   the interface to bind, e.g. {@code "127.0.0.1"} or {@code "0.0.0.0"}
     * @param port   the TCP port to listen on
     * @param router the router that turns requests into responses
     */
    public OaiHttpServer(String host, int port, OaiRouter router) {
        super(host, port);
        this.router = router;
    }

    @Override
    public Response serve(IHTTPSession session) {
        final String method = session.getMethod().name();
        final String uri = session.getUri();

        @Nullable String body = null;
        if (bodyBearing(method)) {
            final Map<String, String> files = new HashMap<>();
            try {
                session.parseBody(files);
            } catch (IOException | ResponseException e) {
                return newFixedLengthResponse(Response.Status.BAD_REQUEST, MIME_JSON, MALFORMED_BODY_JSON);
            }
            // For non-multipart bodies NanoHTTPD stores the raw payload under "postData".
            body = files.get("postData");
        }

        final OaiResponse routed = router.route(method, uri, body);
        return newFixedLengthResponse(statusFor(routed.getStatus()), MIME_JSON, routed.getBody());
    }

    private static boolean bodyBearing(String method) {
        return "POST".equals(method) || "PUT".equals(method) || "PATCH".equals(method);
    }

    private static Response.IStatus statusFor(int code) {
        switch (code) {
            case 200:
                return Response.Status.OK;
            case 400:
                return Response.Status.BAD_REQUEST;
            case 404:
                return Response.Status.NOT_FOUND;
            case 405:
                return Response.Status.METHOD_NOT_ALLOWED;
            default:
                return Response.Status.INTERNAL_ERROR;
        }
    }
}
