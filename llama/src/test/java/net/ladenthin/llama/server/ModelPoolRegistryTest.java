// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Model-free coverage of {@link ModelPool}'s registry resolution + lazy first-request pull (R4). */
public class ModelPoolRegistryTest {

    @TempDir
    Path tempDir;

    private HttpServer stub;
    private int port;
    private final AtomicReference<String> lastPostBody = new AtomicReference<>();

    @BeforeEach
    void startStub() throws IOException {
        stub = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        stub.createContext("/models", exchange -> {
            String method = exchange.getRequestMethod();
            if ("POST".equalsIgnoreCase(method)) {
                byte[] body = exchange.getRequestBody().readAllBytes();
                lastPostBody.set(new String(body, StandardCharsets.UTF_8));
                send(exchange, 200, "{\"status\":\"ok\"}");
            } else {
                send(exchange, 200, "{\"object\":\"list\",\"data\":[{\"id\":\"x\"}]}");
            }
        });
        stub.start();
        port = stub.getAddress().getPort();
    }

    @AfterEach
    void stopStub() {
        if (stub != null) {
            stub.stop(0);
        }
    }

    private static void send(com.sun.net.httpserver.HttpExchange exchange, int code, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.sendResponseHeaders(code, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    /** Downloader that records whether it was invoked and writes a dummy file. */
    static final class RecordingDownloader implements ModelDownloader {
        boolean called;

        @Override
        public Path download(ResolvedModelSource source, Path targetDir, PullProgressListener listener) throws IOException {
            called = true;
            Files.createDirectories(targetDir);
            Path f = targetDir.resolve("downloaded.gguf");
            Files.write(f, "FAKE".getBytes(StandardCharsets.UTF_8));
            return f;
        }
    }

    @Test
    void loadsExistingLocalWithoutPulling() throws IOException {
        Path local = tempDir.resolve("local.gguf");
        Files.write(local, "X".getBytes(StandardCharsets.UTF_8));
        Path registryFile = tempDir.resolve("models.json");
        ModelRegistry registry = new ModelRegistry(registryFile);
        registry.add(new ModelRegistryEntry.Builder("m")
                .localPath(local.toAbsolutePath().toString())
                .sourceUrl("http://example.com/m.gguf")
                .build());

        RecordingDownloader dl = new RecordingDownloader();
        ModelPool pool = new ModelPool("http://127.0.0.1:" + port, registry, false, dl, tempDir.resolve("store"));
        pool.loadModel("m");

        assertFalse(dl.called, "should not pull when the local file exists");
        assertTrue(lastPostBody.get().contains(local.getFileName().toString()));
    }

    @Test
    void lazilyPullsMissingLocal() throws IOException {
        Path registryFile = tempDir.resolve("models.json");
        ModelRegistry registry = new ModelRegistry(registryFile);
        registry.add(new ModelRegistryEntry.Builder("m").sourceUrl("http://example.com/m.gguf").build());

        RecordingDownloader dl = new RecordingDownloader();
        ModelPool pool = new ModelPool("http://127.0.0.1:" + port, registry, false, dl, tempDir.resolve("store"));
        pool.loadModel("m");

        assertTrue(dl.called, "should pull when the local file is absent");
        assertTrue(lastPostBody.get().contains("downloaded.gguf"));
    }

    @Test
    void offlineMissingLocalThrows() throws IOException {
        Path registryFile = tempDir.resolve("models.json");
        ModelRegistry registry = new ModelRegistry(registryFile);
        registry.add(new ModelRegistryEntry.Builder("m").sourceUrl("http://example.com/m.gguf").build());

        ModelPool pool = new ModelPool("http://127.0.0.1:" + port, registry, true, new RecordingDownloader(), tempDir.resolve("store"));
        assertThrows(IllegalStateException.class, () -> pool.loadModel("m"));
    }

    @Test
    void unknownAliasThrows() throws IOException {
        Path registryFile = tempDir.resolve("models.json");
        ModelRegistry registry = new ModelRegistry(registryFile);
        ModelPool pool = new ModelPool("http://127.0.0.1:" + port, registry, false, new RecordingDownloader(), tempDir.resolve("store"));
        assertThrows(IllegalArgumentException.class, () -> pool.loadModel("nope"));
    }

    @Test
    void noRegistryRefusesAliasLoad() {
        ModelPool pool = new ModelPool("http://127.0.0.1:" + port);
        assertThrows(IllegalStateException.class, () -> pool.loadModel("m"));
    }
}
