// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.fail;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.TestConstants;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Integration test for upstream <em>router mode</em> running inside the JVM via
 * {@link NativeServer}: started without a model argument, the server manages models from
 * {@code --models-dir} and serves each from a worker subprocess. In-JVM the upstream worker-spawn
 * (re-exec of the process binary) would relaunch {@code java} with llama-server arguments, so
 * {@link NativeServer#setWorkerCommand(String...)} redirects workers to a fresh JVM running the
 * classic single-model {@code NativeServer} (patch {@code 0008}).
 *
 * <p>Linux-only: worker relaunch and the symlinked models dir are exercised on one platform;
 * router mode itself is upstream functionality, this test pins the embedded wiring.</p>
 */
@ClaudeGenerated(
        purpose = "Prove in-JVM router mode end to end: model listing from --models-dir, "
                + "explicit /models/load spawning a worker JVM through setWorkerCommand, and a "
                + "chat completion proxied to that worker.")
public class RouterModeIntegrationTest extends OpenAiServerTestSupport {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** Generous ceiling for worker-JVM spawn + model load on a cold CI runner. */
    private static final long MODEL_READY_TIMEOUT_MILLIS = 240_000L;

    @TempDir
    static Path modelsDir;

    private static NativeServer server;
    private static int port;
    private static String modelName;

    @BeforeAll
    public static void setup() throws Exception {
        Assumptions.assumeTrue(
                System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("linux"),
                "Router worker-relaunch test runs on Linux only");
        File reasoningModel = new File(TestConstants.REASONING_MODEL_PATH);
        Assumptions.assumeTrue(reasoningModel.exists(), "Reasoning model not found, skipping router test");
        String classpath = System.getProperty("java.class.path", "");
        Assumptions.assumeTrue(
                !classpath.isEmpty() && !classpath.matches(".*\\s.*"),
                "Classpath contains whitespace; worker command cannot carry it");

        // The models dir must contain ONLY the model under test (the CI models/ dir also holds
        // mmproj/vocoder files the router would list); a symlink avoids copying ~400 MB.
        Files.createSymbolicLink(
                modelsDir.resolve(reasoningModel.getName()),
                reasoningModel.getAbsoluteFile().toPath());

        // Workers must relaunch through this library, not through the JVM binary itself.
        String javaBin =
                Paths.get(System.getProperty("java.home"), "bin", "java").toString();
        NativeServer.setWorkerCommand(javaBin, "-cp", classpath, "net.ladenthin.llama.server.NativeServer");

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        port = findFreePort();
        // The router forwards its own base arguments (context size, GPU layers) to each worker.
        server = new NativeServer(
                        "--host",
                        "127.0.0.1",
                        "--port",
                        Integer.toString(port),
                        "--models-dir",
                        modelsDir.toString(),
                        "--models-max",
                        "1",
                        "-c",
                        "512",
                        "-ngl",
                        Integer.toString(gpuLayers))
                .start();
        awaitHttp("/health", 30_000L);

        modelName = discoverModelName();
        loadModelAndAwaitReady();
    }

    @AfterAll
    public static void tearDown() {
        if (server != null) {
            server.close(); // router clean-up unloads (terminates) all worker instances
        }
        NativeServer.setWorkerCommand(); // clear the process-wide override
    }

    private static int findFreePort() throws IOException {
        try (ServerSocket socket = new ServerSocket(0)) {
            return socket.getLocalPort();
        }
    }

    private static void awaitHttp(String path, long timeoutMillis) throws Exception {
        long deadline = System.currentTimeMillis() + timeoutMillis;
        IOException last = null;
        while (System.currentTimeMillis() < deadline) {
            try {
                if (new RouterModeIntegrationTest().get(port, path, "").code == 200) {
                    return;
                }
            } catch (IOException e) {
                last = e;
            }
            Thread.sleep(200L);
        }
        fail("router did not answer " + path + " within " + timeoutMillis + "ms" + (last != null ? ": " + last : ""));
    }

    /** Finds the entry for the symlinked GGUF in GET /models and returns its model identifier. */
    private static String discoverModelName() throws Exception {
        Response models = new RouterModeIntegrationTest().get(port, "/models", "");
        assertThat(models.body, models.code, is(200));
        JsonNode root = MAPPER.readTree(models.body);
        JsonNode data = root.has("data") ? root.get("data") : root.path("models");
        for (JsonNode entry : data) {
            String id = entry.path("id").asText(entry.path("name").asText(""));
            if (id.contains("Qwen3")) {
                return id;
            }
        }
        fail("GET /models did not list the symlinked model: " + models.body);
        return ""; // unreachable
    }

    private static void loadModelAndAwaitReady() throws Exception {
        Response load =
                new RouterModeIntegrationTest().post(port, "/models/load", "{\"model\":\"" + modelName + "\"}", "");
        assertThat(load.body, load.code, is(200));

        long deadline = System.currentTimeMillis() + MODEL_READY_TIMEOUT_MILLIS;
        String lastBody = "";
        while (System.currentTimeMillis() < deadline) {
            Response models = new RouterModeIntegrationTest().get(port, "/models", "");
            lastBody = models.body;
            JsonNode root = MAPPER.readTree(models.body);
            JsonNode data = root.has("data") ? root.get("data") : root.path("models");
            for (JsonNode entry : data) {
                String id = entry.path("id").asText(entry.path("name").asText(""));
                if (id.equals(modelName)) {
                    // Router entries carry {"status": {"value": "unloaded|loading|loaded|..."}}
                    // (server-models.h server_model_status_to_string).
                    String status = entry.path("status").path("value").asText("");
                    if ("loaded".equals(status)) {
                        return;
                    }
                }
            }
            Thread.sleep(2_000L);
        }
        fail("model '" + modelName + "' did not become ready within " + MODEL_READY_TIMEOUT_MILLIS
                + "ms; last GET /models: " + lastBody);
    }

    @Test
    public void chatCompletion_isProxiedToWorker() throws IOException {
        Response response = post(
                port,
                "/v1/chat/completions",
                "{\"model\":\"" + modelName + "\",\"messages\":[{\"role\":\"user\",\"content\":\"Say one word.\"}],"
                        + "\"max_tokens\":8}",
                "");
        assertThat(response.body, response.code, is(200));
        assertThat(response.body, containsString("\"choices\""));
    }

    @Test
    public void models_listContainsLoadedModel() throws IOException {
        Response models = get(port, "/models", "");
        assertThat(models.code, is(200));
        assertThat(models.body, containsString(modelName));
    }
}
