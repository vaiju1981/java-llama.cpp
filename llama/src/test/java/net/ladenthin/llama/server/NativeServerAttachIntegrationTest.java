// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.TestConstants;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Integration test for {@link NativeServer}'s <em>attach</em> mode
 * ({@link NativeServer#NativeServer(net.ladenthin.llama.LlamaModel, String...)}): the full
 * upstream HTTP frontend serving an already-loaded {@link LlamaModel} — one copy of the weights,
 * shared between direct JNI calls and HTTP requests.
 */
@ClaudeGenerated(
        purpose = "Prove NativeServer attach mode end to end: the upstream HTTP frontend serves "
                + "an already-loaded LlamaModel (health/props/completion/chat over HTTP) while "
                + "direct JNI calls on the same model keep working — one copy of the weights.")
public class NativeServerAttachIntegrationTest extends OpenAiServerTestSupport {

    private static LlamaModel model;
    private static NativeServer server;
    private static int port;

    @BeforeAll
    public static void setup() throws Exception {
        Assumptions.assumeTrue(
                new File(TestConstants.REASONING_MODEL_PATH).exists(),
                "Reasoning model not found, skipping NativeServerAttachIntegrationTest");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.REASONING_MODEL_PATH)
                .setCtxSize(1024)
                .setGpuLayers(gpuLayers)
                .setFit(false));
        port = findFreePort();
        server = new NativeServer(model, "--host", "127.0.0.1", "--port", Integer.toString(port)).start();
        awaitHealthy();
    }

    @AfterAll
    public static void tearDown() {
        // Server before model: the attached routes point into the model's native server context.
        if (server != null) {
            server.close();
        }
        if (model != null) {
            model.close();
        }
    }

    private static int findFreePort() throws IOException {
        try (ServerSocket socket = new ServerSocket(0)) {
            return socket.getLocalPort();
        }
    }

    private static void awaitHealthy() throws Exception {
        long deadline = System.currentTimeMillis() + 30_000L;
        IOException last = null;
        while (System.currentTimeMillis() < deadline) {
            try {
                Response health = new NativeServerAttachIntegrationTest().get(port, "/health", "");
                if (health.code == 200) {
                    return;
                }
            } catch (IOException e) {
                last = e;
            }
            Thread.sleep(200L);
        }
        fail("attached server did not become healthy within 30s" + (last != null ? ": " + last : ""));
    }

    @Test
    public void health_isOk() throws IOException {
        assertThat(get(port, "/health", "").code, is(200));
    }

    /** The attached frontend reads model metadata straight from the shared, loaded context. */
    @Test
    public void props_reportModelContext() throws IOException {
        Response props = get(port, "/props", "");
        assertThat(props.code, is(200));
        assertThat(props.body, containsString("default_generation_settings"));
    }

    @Test
    public void completion_overHttp_served() throws IOException {
        Response response = post(port, "/completion", "{\"prompt\":\"Hello\",\"n_predict\":4,\"temperature\":0}", "");
        assertThat(response.body, response.code, is(200));
        assertThat(response.body, containsString("\"content\""));
    }

    @Test
    public void chatCompletion_overHttp_served() throws IOException {
        Response response = post(
                port,
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"Say one word.\"}],\"max_tokens\":8}",
                "");
        assertThat(response.body, response.code, is(200));
        assertThat(response.body, containsString("\"choices\""));
    }

    /** The model stays fully usable for direct JNI calls while the HTTP frontend is attached. */
    @Test
    public void directJniCalls_stillWork_whileAttached() {
        String completion =
                model.complete(new InferenceParameters("2+2=").withNPredict(4).withTemperature(0.0f));
        assertNotNull(completion);
    }
}
