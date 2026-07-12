// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Model-free coverage of {@link ModelPool}: it issues the right upstream calls and parses results. */
public class ModelPoolTest {

    private HttpServer stub;
    private int port;

    @BeforeEach
    void startStub() throws IOException {
        stub = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        stub.createContext("/models", exchange -> {
            String method = exchange.getRequestMethod();
            if ("GET".equalsIgnoreCase(method)) {
                send(exchange, 200, "{\"object\":\"list\",\"data\":[{\"id\":\"a\"},{\"id\":\"b\"}]}");
            } else if ("POST".equalsIgnoreCase(method)) {
                send(exchange, 200, "{\"status\":\"ok\"}");
            } else if ("DELETE".equalsIgnoreCase(method)) {
                send(exchange, 200, "{\"status\":\"unloaded\"}");
            } else {
                send(exchange, 405, "{\"error\":\"method\"}");
            }
        });
        stub.createContext("/health", exchange -> send(exchange, 200, "{\"status\":\"ok\"}"));
        stub.createContext(
                "/metrics",
                exchange -> send(exchange, 200, "{\"idle\":2,\"processing\":1,\"deferred\":3,\"slots\":[]}"));
        stub.start();
        port = stub.getAddress().getPort();
    }

    @AfterEach
    void stopStub() {
        if (stub != null) {
            stub.stop(0);
        }
    }

    @Test
    void listModelsParsesDataArray() throws Exception {
        ModelPool pool = new ModelPool("http://127.0.0.1:" + port);
        assertEquals(java.util.List.of("a", "b"), pool.listModels());
    }

    @Test
    void loadAndUnloadSucceed() throws Exception {
        ModelPool pool = new ModelPool("http://127.0.0.1:" + port);
        pool.loadModel("a", new ModelParameters().setModel("models/a.gguf"));
        pool.unloadModel("a");
    }

    @Test
    void healthProbeReportsHealthy() throws Exception {
        ModelPool pool = new ModelPool("http://127.0.0.1:" + port);
        ModelHealth health = pool.getModelHealth("a");
        assertTrue(health.healthy());
        assertTrue(health.latencyMillis() >= 0);
    }

    @Test
    void metricsParsed() throws Exception {
        ModelPool pool = new ModelPool("http://127.0.0.1:" + port);
        assertEquals(2, pool.getModelMetrics("a").getIdleSlots());
    }

    @Test
    void nonOkResponseThrows() throws Exception {
        // A DELETE to a missing model returns 404 from a stricter server; assert the failure path
        // surfaces as IOException by pointing at a context that always 404s.
        HttpServer failing = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        failing.createContext("/models", exchange -> send(exchange, 404, "{\"error\":\"not found\"}"));
        failing.start();
        try {
            ModelPool pool =
                    new ModelPool("http://127.0.0.1:" + failing.getAddress().getPort());
            boolean threw = false;
            try {
                pool.unloadModel("missing");
            } catch (IOException e) {
                threw = true;
            }
            assertTrue(threw, "unloadModel should throw on non-2xx");
        } finally {
            failing.stop(0);
        }
    }

    private static void send(com.sun.net.httpserver.HttpExchange exchange, int code, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(code, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }
}
