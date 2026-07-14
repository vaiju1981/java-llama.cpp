// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpServer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Model-free coverage of {@link ModelRegistryCli}: list/show/rm/cp and pull against a stub server. */
public class ModelRegistryCliTest {

    @TempDir
    Path tempDir;

    private HttpServer stub;
    private int port;

    @BeforeEach
    void startStub() throws IOException {
        stub = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        byte[] body = "GGUF".getBytes(StandardCharsets.UTF_8);
        stub.createContext("/m.gguf", exchange -> {
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

    private Path registryFile() {
        return tempDir.resolve("models.json");
    }

    private String[] withRegistry(String... args) {
        String[] out = new String[args.length + 2];
        out[0] = args[0];
        out[1] = "--registry";
        out[2] = registryFile().toString();
        System.arraycopy(args, 1, out, 3, args.length - 1);
        return out;
    }

    private static String runOut(String[] args) {
        ByteArrayOutputStream buf = new ByteArrayOutputStream();
        int code = ModelRegistryCli.run(args, new PrintStream(buf), new PrintStream(new ByteArrayOutputStream()));
        return code + "\u0000" + buf.toString(StandardCharsets.UTF_8);
    }

    @Test
    void listEmptyThenAfterPull() throws IOException {
        // Seed the registry with one entry via the API, then list through the CLI.
        ModelRegistry registry = new ModelRegistry(registryFile());
        registry.add(new ModelRegistryEntry.Builder("seed")
                .localPath("/x/seed.gguf")
                .quantization("Q4_K_M")
                .sizeBytes(42L)
                .pulledAt(9L)
                .build());

        String res = runOut(withRegistry("list"));
        assertEquals(0, Integer.parseInt(res.substring(0, res.indexOf('\u0000'))));
        assertTrue(res.contains("seed"));
        assertTrue(res.contains("Q4_K_M"));
    }

    @Test
    void showAndRemove() throws IOException {
        ModelRegistry registry = new ModelRegistry(registryFile());
        registry.add(new ModelRegistryEntry.Builder("foo").localPath("/x/foo.gguf").build());

        String shown = runOut(withRegistry("show", "foo"));
        assertTrue(shown.contains("\"name\" : \"foo\"") || shown.contains("\"name\":\"foo\"") || shown.contains("foo"));

        String removed = runOut(withRegistry("rm", "foo"));
        assertTrue(removed.contains("Removed 'foo'"));
        assertEquals(0, new ModelRegistry(registryFile()).size());
    }

    @Test
    void copyCreatesAlias() throws IOException {
        ModelRegistry registry = new ModelRegistry(registryFile());
        registry.add(new ModelRegistryEntry.Builder("src")
                .localPath("/x/src.gguf")
                .quantization("Q8_0")
                .build());

        String copied = runOut(withRegistry("cp", "src", "dst"));
        assertTrue(copied.contains("Copied 'src' -> 'dst'"));
        ModelRegistry reloaded = new ModelRegistry(registryFile());
        assertTrue(reloaded.contains("dst"));
        assertEquals("/x/src.gguf", reloaded.get("dst").getLocalPath());
    }

    @Test
    void pullDownloadsAndRegisters() throws IOException {
        String[] args = withRegistry(
                "pull", "http://127.0.0.1:" + port + "/m.gguf", "--models-dir", tempDir.resolve("store").toString());
        String res = runOut(args);
        assertEquals(0, Integer.parseInt(res.substring(0, res.indexOf('\u0000'))));
        assertTrue(res.contains("Pulled"));
        ModelRegistry reloaded = new ModelRegistry(registryFile());
        assertTrue(reloaded.contains("m"));
        assertTrue(Files.exists(java.nio.file.Paths.get(reloaded.get("m").getLocalPath())));
    }

    @Test
    void unknownCommandReturnsUsageExit() {
        String res = runOut(new String[] {"bogus"});
        assertEquals(2, Integer.parseInt(res.substring(0, res.indexOf('\u0000'))));
    }
}
