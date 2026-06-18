// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.nio.file.Path;
import java.nio.file.Paths;
import org.jspecify.annotations.Nullable;

/**
 * Command-line argument parser for {@link LlamaServer}. Pure and free of any native dependency, so
 * it can be unit-tested in isolation (no socket, no model).
 *
 * <p>{@link #parse(String[])} returns a validated {@link LlamaServerConfig} or throws
 * {@link IllegalArgumentException} (whose message includes the {@link #usage()} text) for unknown
 * flags, missing values or a missing required {@code --model}. {@code -h}/{@code --help} is detected
 * separately via {@link #isHelpRequested(String[])} so callers can print help without it being
 * treated as an error.</p>
 */
public final class LlamaServerArgs {

    /** Default bind interface (loopback only; pass {@code --host 0.0.0.0} to expose on the LAN). */
    public static final String DEFAULT_HOST = "127.0.0.1";

    /** Default TCP port. */
    public static final int DEFAULT_PORT = 8080;

    private LlamaServerArgs() {}

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
     * Parse the command-line arguments into a {@link LlamaServerConfig}.
     *
     * @param args the raw command-line arguments
     * @return the validated configuration
     * @throws IllegalArgumentException if an argument is unknown, a value is missing or malformed,
     *                                  or the required {@code --model} is absent
     */
    public static LlamaServerConfig parse(String... args) {
        String host = DEFAULT_HOST;
        int port = DEFAULT_PORT;
        @Nullable String modelPath = null;
        @Nullable String modelAlias = null;
        int ctxSize = 0;
        int gpuLayers = 0;
        int threads = 0;
        boolean embedding = false;

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
                    ctxSize = intValue(args, ++i, arg);
                    break;
                case "-ngl":
                case "--n-gpu-layers":
                    gpuLayers = intValue(args, ++i, arg);
                    break;
                case "-t":
                case "--threads":
                    threads = intValue(args, ++i, arg);
                    break;
                case "--model-alias":
                    modelAlias = nextValue(args, ++i, arg);
                    break;
                case "--embedding":
                case "--embeddings":
                    embedding = true;
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
        final String alias = modelAlias != null ? modelAlias : deriveAlias(modelPath);
        return new LlamaServerConfig(host, port, modelPath, alias, ctxSize, gpuLayers, threads, embedding);
    }

    /**
     * The human-readable usage / help text.
     *
     * @return the usage text
     */
    public static String usage() {
        return String.join(
                System.lineSeparator(),
                "LlamaServer - OpenAI-compatible HTTP server for java-llama.cpp",
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
                "  --model-alias <name>       Model id reported by /v1/models (default: file name)",
                "  --embedding                Load in embedding mode (enables POST /v1/embeddings)",
                "  -h,  --help                Show this help and exit",
                "",
                "Endpoints:",
                "  POST /v1/chat/completions",
                "  POST /v1/completions",
                "  POST /v1/embeddings        (requires --embedding)",
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

    private static String deriveAlias(String modelPath) {
        final Path name = Paths.get(modelPath).getFileName();
        return name != null ? name.toString() : modelPath;
    }

    private static IllegalArgumentException error(String message) {
        return error(message, null);
    }

    private static IllegalArgumentException error(String message, @Nullable Throwable cause) {
        return new IllegalArgumentException(message + System.lineSeparator() + System.lineSeparator() + usage(), cause);
    }
}
