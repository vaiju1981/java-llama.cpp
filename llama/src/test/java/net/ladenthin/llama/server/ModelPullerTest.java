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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Model-free coverage of {@link ModelPuller}: download + register against a stub HTTP server. */
public class ModelPullerTest {

    @TempDir
    Path tempDir;

    private HttpServer stub;
    private int port;

    @BeforeEach
    void startStub() throws IOException {
        stub = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        byte[] body = "GGUF-FAKE-BYTES".getBytes(StandardCharsets.UTF_8);
        stub.createContext("/model.gguf", exchange -> {
            exchange.getResponseHeaders().set("Content-Type", "application/octet-stream");
            exchange.sendResponseHeaders(200, body.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(body);
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

    private ModelPuller puller(boolean offline) throws IOException {
        ModelRegistry registry = new ModelRegistry(tempDir.resolve("models.json"));
        return new ModelPuller(
                registry, new ModelNameResolver(), new HttpModelDownloader(), tempDir.resolve("models"), offline);
    }

    @Test
    void pullDownloadsAndRegisters() throws IOException {
        ModelPuller puller = puller(false);
        ModelRegistryEntry entry = puller.pull("http://127.0.0.1:" + port + "/model.gguf");

        assertEquals("model", entry.getName());
        assertTrue(entry.getLocalPath() != null && Files.exists(java.nio.file.Paths.get(entry.getLocalPath())));
        assertEquals("http://127.0.0.1:" + port + "/model.gguf", entry.getSourceUrl());
        assertEquals("GGUF-FAKE-BYTES".length(), entry.getSizeBytes());
        assertTrue(entry.getPulledAt() > 0);

        // Persisted to the manifest.
        ModelRegistry reloaded = new ModelRegistry(tempDir.resolve("models.json"));
        assertTrue(reloaded.contains("model"));
        assertEquals(entry.getLocalPath(), reloaded.get("model").getLocalPath());
    }

    @Test
    void pullReportsProgress() throws IOException {
        ModelPuller puller = puller(false);
        long[] last = {-1};
        puller.pull("http://127.0.0.1:" + port + "/model.gguf", (bytes, total) -> last[0] = bytes);
        assertTrue(last[0] >= "GGUF-FAKE-BYTES".length());
    }

    @Test
    void offlinePullThrows() throws IOException {
        ModelPuller puller = puller(true);
        assertThrows(IllegalStateException.class, () -> puller.pull("http://127.0.0.1:" + port + "/model.gguf"));
    }

    @Test
    void pullLocalRegistersWithoutDownload() throws IOException {
        Path local = tempDir.resolve("local.gguf");
        Files.write(local, "LOCAL".getBytes(StandardCharsets.UTF_8));
        ModelPuller puller = puller(false);
        ModelRegistryEntry entry = puller.pull(local.toAbsolutePath().toString());
        assertEquals("local", entry.getName());
        assertEquals(local.toAbsolutePath().toString(), entry.getLocalPath());
        assertFalse(puller.isOffline());
    }
}
