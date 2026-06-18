// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

import fi.iki.elonen.NanoHTTPD;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose =
                "End-to-end exercise of OaiHttpServer over a real loopback socket (ephemeral port, fake backend, no "
                        + "native model): confirms the NanoHTTPD adapter extracts the method/URI, reads the JSON POST body via "
                        + "the 'postData' idiom, forwards it to the router, and maps the routed status/body back to the client.")
public class OaiHttpServerIntegrationTest {

    /** Fake backend that echoes the received chat body so the test can assert it round-tripped. */
    private static final class EchoBackend implements OaiBackend {
        private String lastChatBody = "";

        @Override
        public String chatCompletions(String requestJson) {
            lastChatBody = requestJson;
            return "{\"object\":\"chat.completion\",\"echo\":" + requestJson + "}";
        }

        @Override
        public String completions(String requestJson) {
            return "{\"object\":\"text_completion\"}";
        }

        @Override
        public String embeddings(String requestJson) {
            return "{\"object\":\"list\"}";
        }

        @Override
        public String listModels() {
            return "{\"object\":\"list\",\"data\":[]}";
        }

        String lastChatBody() {
            return lastChatBody;
        }
    }

    @Test
    public void servesHealthAndChatOverRealSocket() throws IOException {
        EchoBackend backend = new EchoBackend();
        OaiHttpServer server = new OaiHttpServer("127.0.0.1", 0, new OaiRouter(backend));
        // daemon=true so a failed assertion never leaves a non-daemon listener thread behind.
        server.start(NanoHTTPD.SOCKET_READ_TIMEOUT, true);
        try {
            final int port = server.getListeningPort();
            final String base = "http://127.0.0.1:" + port;

            Response health = httpGet(base + "/health");
            assertThat(health.status, is(200));
            assertThat(health.body, containsString("\"status\":\"ok\""));

            final String chatRequest = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response chat = httpPost(base + "/v1/chat/completions", chatRequest);
            assertThat(chat.status, is(200));
            assertThat(chat.body, containsString("chat.completion"));
            // The JSON POST body reached the backend intact (validates the parseBody/postData path).
            assertThat(backend.lastChatBody(), is(chatRequest));

            Response notFound = httpGet(base + "/v1/nope");
            assertThat(notFound.status, is(404));
        } finally {
            server.stop();
        }
    }

    private static final class Response {
        private final int status;
        private final String body;

        Response(int status, String body) {
            this.status = status;
            this.body = body;
        }
    }

    private static Response httpGet(String url) throws IOException {
        final HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setRequestMethod("GET");
        return readResponse(conn);
    }

    private static Response httpPost(String url, String body) throws IOException {
        final HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setRequestMethod("POST");
        conn.setDoOutput(true);
        conn.setRequestProperty("Content-Type", "application/json");
        try (OutputStream os = conn.getOutputStream()) {
            os.write(body.getBytes(StandardCharsets.UTF_8));
        }
        return readResponse(conn);
    }

    private static Response readResponse(HttpURLConnection conn) throws IOException {
        final int status = conn.getResponseCode();
        try (InputStream in = status < 400 ? conn.getInputStream() : conn.getErrorStream()) {
            final ByteArrayOutputStream out = new ByteArrayOutputStream();
            final byte[] buffer = new byte[1024];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
            return new Response(status, new String(out.toByteArray(), StandardCharsets.UTF_8));
        } finally {
            conn.disconnect();
        }
    }
}
