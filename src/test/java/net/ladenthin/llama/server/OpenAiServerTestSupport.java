// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * Shared HTTP plumbing for {@link OpenAiCompatServer} tests: tiny helpers that POST/GET against a
 * server on {@code 127.0.0.1:<port>} and capture the status code and body.
 *
 * <p>Abstract (and not named {@code *Test}) so the harness never runs it on its own; subclasses
 * supply their own fixtures and assertions — {@link OpenAiCompatServerHttpTest} drives a fake backend,
 * and {@code OpenAiCompatServerIntegrationTest} drives a real model.
 */
abstract class OpenAiServerTestSupport {

    /**
     * POST a JSON body to {@code path}.
     *
     * @param port the server port
     * @param path the request path (e.g. {@code /v1/chat/completions})
     * @param body the JSON request body
     * @param auth an {@code Authorization} header value, or {@code ""} to send none
     * @return the captured response
     * @throws IOException on transport failure
     */
    Response post(int port, String path, String body, String auth) throws IOException {
        HttpURLConnection conn = open(port, path, auth);
        conn.setRequestMethod("POST");
        conn.setDoOutput(true);
        conn.setRequestProperty("Content-Type", "application/json");
        try (OutputStream os = conn.getOutputStream()) {
            os.write(body.getBytes(UTF_8));
        }
        return read(conn);
    }

    /**
     * GET {@code path}.
     *
     * @param port the server port
     * @param path the request path
     * @param auth an {@code Authorization} header value, or {@code ""} to send none
     * @return the captured response
     * @throws IOException on transport failure
     */
    Response get(int port, String path, String auth) throws IOException {
        HttpURLConnection conn = open(port, path, auth);
        conn.setRequestMethod("GET");
        return read(conn);
    }

    private static HttpURLConnection open(int port, String path, String auth) throws IOException {
        HttpURLConnection conn = (HttpURLConnection) new URL("http://127.0.0.1:" + port + path).openConnection();
        if (!auth.isEmpty()) {
            conn.setRequestProperty("Authorization", auth);
        }
        return conn;
    }

    private static Response read(HttpURLConnection conn) throws IOException {
        int code = conn.getResponseCode();
        InputStream is = code < 400 ? conn.getInputStream() : conn.getErrorStream();
        String body = is == null ? "" : readAll(is);
        return new Response(code, body);
    }

    private static String readAll(InputStream is) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] chunk = new byte[1024];
        int read;
        while ((read = is.read(chunk)) != -1) {
            buffer.write(chunk, 0, read);
        }
        return new String(buffer.toByteArray(), UTF_8);
    }

    /** Captured HTTP response: status code and body text. */
    static final class Response {
        final int code;
        final String body;

        Response(int code, String body) {
            this.code = code;
            this.body = body;
        }
    }
}
