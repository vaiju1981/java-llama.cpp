// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.nio.file.Path;
import java.nio.file.Paths;
import net.ladenthin.llama.parameters.ModelParameters;
import org.jspecify.annotations.Nullable;

/**
 * Command-line argument parser for {@link OpenAiCompatServer}'s standalone launcher. Pure and free of
 * any native dependency, so it can be unit-tested in isolation (no socket, no model).
 *
 * <p>{@link #parse(String[])} returns an immutable {@link Options} or throws
 * {@link IllegalArgumentException} (whose message embeds the {@link #usage()} text) for unknown flags,
 * missing values or a missing required {@code --model}. {@code -h}/{@code --help} is detected separately
 * via {@link #isHelpRequested(String[])} so callers can print help without it being treated as an error.
 *
 * <p>Flags mirror llama.cpp's own server where they overlap ({@code -m}, {@code -p}, {@code -c},
 * {@code -ngl}, {@code -t}); a few legacy spellings are accepted as aliases so earlier documented
 * invocations keep working.
 */
public final class OpenAiServerCli {

    /** Default bind interface (loopback only; pass {@code --host 0.0.0.0} to expose on the LAN). */
    public static final String DEFAULT_HOST = OpenAiServerConfig.DEFAULT_HOST;

    /** Default TCP port. */
    public static final int DEFAULT_PORT = OpenAiServerConfig.DEFAULT_PORT;

    private OpenAiServerCli() {}

    /**
     * Whether the arguments request the help text.
     *
     * @param args the raw command-line arguments
     * @return {@code true} if {@code -h} or {@code --help} is present
     */
    public static boolean isHelpRequested(String... args) {
        for (final String arg : args) {
            if ("-h".equals(arg) || "--help".equals(arg)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Parse the command-line arguments into validated {@link Options}.
     *
     * @param args the raw command-line arguments
     * @return the parsed options
     * @throws IllegalArgumentException if an argument is unknown, a value is missing or malformed,
     *                                  or the required {@code --model} is absent
     */
    public static Options parse(String... args) {
        String host = DEFAULT_HOST;
        int port = DEFAULT_PORT;
        @Nullable String modelPath = null;
        @Nullable String modelId = null;
        @Nullable String apiKey = null;
        @Nullable String mmproj = null;
        int ctxSize = 0;
        int gpuLayers = 0;
        int threads = 0;
        int parallel = 0;
        boolean embedding = false;
        boolean reranking = false;

        for (int i = 0; i < args.length; i++) {
            final String arg = args[i];
            switch (arg) {
                case "-m":
                case "--model":
                    modelPath = nextValue(args, ++i, arg);
                    break;
                case "--host":
                    host = nextValue(args, ++i, arg);
                    break;
                case "-p":
                case "--port":
                    port = intValue(args, ++i, arg);
                    break;
                case "-c":
                case "--ctx-size":
                case "--ctx":
                    ctxSize = intValue(args, ++i, arg);
                    break;
                case "-ngl":
                case "--n-gpu-layers":
                case "--gpu-layers":
                    gpuLayers = intValue(args, ++i, arg);
                    break;
                case "-t":
                case "--threads":
                    threads = intValue(args, ++i, arg);
                    break;
                case "--parallel":
                    parallel = intValue(args, ++i, arg);
                    break;
                case "--model-id":
                case "--model-alias":
                    modelId = nextValue(args, ++i, arg);
                    break;
                case "--api-key":
                    apiKey = nextValue(args, ++i, arg);
                    break;
                case "--mmproj":
                    mmproj = nextValue(args, ++i, arg);
                    break;
                case "--embedding":
                case "--embeddings":
                    embedding = true;
                    break;
                case "--reranking":
                case "--rerank":
                    reranking = true;
                    break;
                case "-h":
                case "--help":
                    // Detected by isHelpRequested(); accepted here so parse() still succeeds.
                    break;
                default:
                    throw error("Unknown argument: " + arg);
            }
        }

        if (modelPath == null) {
            throw error("Missing required argument: -m/--model <path-to-gguf>");
        }
        return new Options(
                host, port, modelPath, modelId, apiKey, mmproj, ctxSize, gpuLayers, threads, parallel, embedding,
                reranking);
    }

    /**
     * The human-readable usage / help text.
     *
     * @return the usage text
     */
    public static String usage() {
        return String.join(
                System.lineSeparator(),
                "OpenAiCompatServer - OpenAI-compatible HTTP server for java-llama.cpp",
                "",
                "Usage:",
                "  java -jar llama-<version>-jar-with-dependencies.jar --model <path.gguf> [options]",
                "",
                "Required:",
                "  -m,  --model <path>        Path to the GGUF model file",
                "",
                "Options:",
                "  --host <host>              Interface to bind (default: " + DEFAULT_HOST + ")",
                "  -p,  --port <port>         TCP port to listen on (default: " + DEFAULT_PORT + ")",
                "  -c,  --ctx-size <n>        Context window size (default: llama.cpp default)",
                "  -ngl,--n-gpu-layers <n>    Layers to offload to GPU (default: 0 = CPU only)",
                "  -t,  --threads <n>         Inference thread count (default: llama.cpp default)",
                "  --parallel <n>             Parallel inference slots (default: llama.cpp default)",
                "  --model-id <name>          Model id reported by /v1/models (default: file name)",
                "  --api-key <key>            Require an 'Authorization: Bearer <key>' header",
                "  --mmproj <path>            Multimodal projector for vision models (enables image input)",
                "  --embedding                Load in embedding mode (enables POST /v1/embeddings)",
                "  --reranking                Load in reranking mode (enables POST /v1/rerank)",
                "  -h,  --help                Show this help and exit",
                "",
                "Endpoints:",
                "  POST /v1/chat/completions  (streaming via SSE + non-streaming)",
                "  POST /v1/completions",
                "  POST /v1/embeddings        (requires --embedding)",
                "  POST /v1/rerank            (requires --reranking)",
                "  POST /infill               (fill-in-the-middle / autocomplete)",
                "  GET  /v1/models",
                "  GET  /health");
    }

    private static String nextValue(String[] args, int valueIndex, String flag) {
        if (valueIndex >= args.length) {
            throw error("Missing value for " + flag);
        }
        return args[valueIndex];
    }

    private static int intValue(String[] args, int valueIndex, String flag) {
        final String raw = nextValue(args, valueIndex, flag);
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException e) {
            throw error(flag + " expects an integer, got: " + raw, e);
        }
    }

    private static IllegalArgumentException error(String message) {
        return error(message, null);
    }

    private static IllegalArgumentException error(String message, @Nullable Throwable cause) {
        return new IllegalArgumentException(message + System.lineSeparator() + System.lineSeparator() + usage(), cause);
    }

    /**
     * Immutable, parsed launcher options. {@code ctxSize}, {@code threads} and {@code parallel} use
     * {@code 0} as a sentinel meaning "leave the llama.cpp default" — they are only applied to
     * {@link ModelParameters} when positive. {@code gpuLayers} is always applied (its own default of
     * {@code 0} already means CPU-only).
     */
    public static final class Options {

        private final String host;
        private final int port;
        private final String modelPath;
        private final @Nullable String modelId;
        private final @Nullable String apiKey;
        private final @Nullable String mmproj;
        private final int ctxSize;
        private final int gpuLayers;
        private final int threads;
        private final int parallel;
        private final boolean embedding;
        private final boolean reranking;

        private Options(
                String host,
                int port,
                String modelPath,
                @Nullable String modelId,
                @Nullable String apiKey,
                @Nullable String mmproj,
                int ctxSize,
                int gpuLayers,
                int threads,
                int parallel,
                boolean embedding,
                boolean reranking) {
            this.host = host;
            this.port = port;
            this.modelPath = modelPath;
            this.modelId = modelId;
            this.apiKey = apiKey;
            this.mmproj = mmproj;
            this.ctxSize = ctxSize;
            this.gpuLayers = gpuLayers;
            this.threads = threads;
            this.parallel = parallel;
            this.embedding = embedding;
            this.reranking = reranking;
        }

        /**
         * The interface to bind.
         *
         * @return the bind host
         */
        public String getHost() {
            return host;
        }

        /**
         * The TCP port to listen on.
         *
         * @return the port
         */
        public int getPort() {
            return port;
        }

        /**
         * The path to the GGUF model file to load.
         *
         * @return the model path
         */
        public String getModelPath() {
            return modelPath;
        }

        /**
         * The advertised model id, resolved from {@code --model-id} or derived from the model file name.
         *
         * @return the model id reported by {@code GET /v1/models}
         */
        public String getModelId() {
            if (modelId != null) {
                return modelId;
            }
            final Path name = Paths.get(modelPath).getFileName();
            return name != null ? name.toString() : OpenAiServerConfig.DEFAULT_MODEL_ID;
        }

        /**
         * The optional bearer API key.
         *
         * @return the API key, or {@code null} when authentication is disabled
         */
        public @Nullable String getApiKey() {
            return apiKey;
        }

        /**
         * The optional multimodal projector path for vision models.
         *
         * @return the mmproj path, or {@code null} when no vision projector is configured
         */
        public @Nullable String getMmproj() {
            return mmproj;
        }

        /**
         * The context window size, or {@code 0} for the llama.cpp default.
         *
         * @return the context size
         */
        public int getCtxSize() {
            return ctxSize;
        }

        /**
         * The number of layers to offload to the GPU ({@code 0} = CPU-only).
         *
         * @return the GPU layer count
         */
        public int getGpuLayers() {
            return gpuLayers;
        }

        /**
         * The inference thread count, or {@code 0} for the llama.cpp default.
         *
         * @return the thread count
         */
        public int getThreads() {
            return threads;
        }

        /**
         * The number of parallel inference slots, or {@code 0} for the llama.cpp default.
         *
         * @return the parallel slot count
         */
        public int getParallel() {
            return parallel;
        }

        /**
         * Whether to load the model in embedding mode.
         *
         * @return {@code true} if embedding mode is requested
         */
        public boolean isEmbedding() {
            return embedding;
        }

        /**
         * Whether to load the model in reranking mode.
         *
         * @return {@code true} if reranking mode is requested
         */
        public boolean isReranking() {
            return reranking;
        }

        /**
         * Build the {@link ModelParameters} for loading the model described by these options.
         *
         * @return the model parameters
         */
        public ModelParameters toModelParameters() {
            final ModelParameters params =
                    new ModelParameters().setModel(modelPath).setGpuLayers(gpuLayers);
            if (mmproj != null) {
                params.setMmproj(mmproj);
            }
            if (ctxSize > 0) {
                params.setCtxSize(ctxSize);
            }
            if (threads > 0) {
                params.setThreads(threads);
            }
            if (parallel > 0) {
                params.setParallel(parallel);
            }
            if (embedding) {
                params.enableEmbedding();
            }
            if (reranking) {
                params.enableReranking();
            }
            return params;
        }

        /**
         * Build the {@link OpenAiServerConfig} describing the server side of these options. When a
         * context size is given, the advertised input/output token budgets are derived from it.
         *
         * @return the server configuration
         */
        public OpenAiServerConfig toServerConfig() {
            final OpenAiServerConfig.Builder builder = OpenAiServerConfig.builder()
                    .host(host)
                    .port(port)
                    .modelId(getModelId())
                    .supportsVision(mmproj != null);
            if (apiKey != null) {
                builder.apiKey(apiKey);
            }
            if (ctxSize > 0) {
                builder.maxOutputTokens(
                        Math.min(OpenAiServerConfig.DEFAULT_MAX_OUTPUT_TOKENS, Math.max(1, ctxSize / 2)));
                builder.maxInputTokens(Math.max(1, ctxSize - OpenAiServerConfig.DEFAULT_MAX_OUTPUT_TOKENS));
            }
            return builder.build();
        }
    }
}
