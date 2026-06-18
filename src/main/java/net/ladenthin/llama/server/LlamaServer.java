// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import fi.iki.elonen.NanoHTTPD;
import java.io.IOException;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.parameters.ModelParameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Entry point for the optional OpenAI-compatible HTTP server, and the {@code Main-Class} of the
 * {@code -jar-with-dependencies} assembly.
 *
 * <p>It parses the command line ({@link LlamaServerArgs}), loads a GGUF model into a
 * {@link LlamaModel}, and serves OpenAI-compatible endpoints over NanoHTTPD via {@link OaiRouter} /
 * {@link OaiHttpServer}. A shutdown hook stops the server and closes the model on JVM exit
 * (e.g. Ctrl-C / SIGTERM). Run {@code --help} for the full option list.</p>
 *
 * <p>Example:</p>
 *
 * <pre>{@code
 * java -jar llama-<version>-jar-with-dependencies.jar \
 *     --model models/Qwen3-0.6B-Q4_K_M.gguf --host 0.0.0.0 --port 8080 --n-gpu-layers 99
 * }</pre>
 *
 * <p>Responses are non-streaming: the full JSON result is returned per request.</p>
 */
public final class LlamaServer {

    private static final Logger LOG = LoggerFactory.getLogger(LlamaServer.class);

    private LlamaServer() {}

    /**
     * Start the server (blocks the JVM alive on a non-daemon listener thread), or print help.
     *
     * @param args command-line arguments; see {@link LlamaServerArgs#usage()}
     * @throws IOException if the HTTP server cannot bind the configured host/port
     */
    public static void main(String[] args) throws IOException {
        if (LlamaServerArgs.isHelpRequested(args)) {
            LOG.info("{}{}", System.lineSeparator(), LlamaServerArgs.usage());
            return;
        }

        final LlamaServerConfig config = LlamaServerArgs.parse(args);
        final LlamaModel model = loadModel(config);
        final OaiBackend backend = new LlamaModelOaiBackend(model, config.getModelAlias());
        final OaiHttpServer server = new OaiHttpServer(config.getHost(), config.getPort(), new OaiRouter(backend));

        Runtime.getRuntime().addShutdownHook(new Thread(() -> shutdown(server, model), "llama-server-shutdown"));

        try {
            // daemon=false: the non-daemon listener thread keeps the JVM alive after main() returns.
            server.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false);
        } catch (IOException e) {
            // Close the just-loaded native model before propagating the bind failure.
            model.close();
            throw e;
        }

        LOG.info(
                "LlamaServer listening on http://{}:{} (model={})",
                config.getHost(),
                config.getPort(),
                config.getModelAlias());
    }

    private static LlamaModel loadModel(LlamaServerConfig config) {
        final ModelParameters params =
                new ModelParameters().setModel(config.getModelPath()).setGpuLayers(config.getGpuLayers());
        if (config.getCtxSize() > 0) {
            params.setCtxSize(config.getCtxSize());
        }
        if (config.getThreads() > 0) {
            params.setThreads(config.getThreads());
        }
        if (config.isEmbedding()) {
            params.enableEmbedding();
        }
        LOG.info("Loading model {} ...", config.getModelPath());
        return new LlamaModel(params);
    }

    private static void shutdown(OaiHttpServer server, LlamaModel model) {
        server.stop();
        model.close();
    }
}
