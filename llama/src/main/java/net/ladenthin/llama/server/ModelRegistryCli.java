// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import org.jspecify.annotations.Nullable;

/**
 * Command-line front-end for the local model registry — the Ollama-like {@code pull} / {@code list} /
 * {@code show} / {@code rm} / {@code cp} verbs. Thin wrapper over {@link ModelRegistry} and
 * {@link ModelPuller}; no native dependency, so it is model-free and unit-testable.
 *
 * <p>Global {@code --registry <path>} selects the manifest file (otherwise
 * {@link ModelRegistry#DEFAULT_PATH} / the {@value ModelRegistry#PROP_PATH} system property).
 * {@code pull} additionally accepts {@code --offline} and {@code --models-dir <dir>}.
 */
public final class ModelRegistryCli {

    private ModelRegistryCli() {}

    /** Process exit code for a successful invocation. */
    public static final int EXIT_OK = 0;

    /** Process exit code for a usage / unknown-command error. */
    public static final int EXIT_USAGE = 2;

    /** Process exit code for a runtime failure (IO, pull error). */
    public static final int EXIT_ERROR = 1;

    /**
     * CLI entry point.
     *
     * @param args the command-line arguments
     */
    public static void main(String[] args) {
        int code = run(args, System.out, System.err);
        if (code != EXIT_OK) {
            System.exit(code);
        }
    }

    /**
     * Execute a CLI invocation against the given streams (no {@code System.exit}).
     *
     * @param args the command-line arguments
     * @param out the output stream (stdout)
     * @param err the error stream (stderr)
     * @return the process exit code
     */
    public static int run(String[] args, PrintStream out, PrintStream err) {
        if (args.length == 0) {
            printUsage(out);
            return EXIT_OK;
        }
        String cmd = args[0];
        try {
            switch (cmd) {
                case "pull":
                    return cmdPull(args, out);
                case "list":
                    return cmdList(args, out);
                case "show":
                    return cmdShow(args, out, err);
                case "rm":
                    return cmdRemove(args, out, err);
                case "cp":
                    return cmdCopy(args, out, err);
                case "-h":
                case "--help":
                    printUsage(out);
                    return EXIT_OK;
                default:
                    err.println("Unknown command: " + cmd);
                    printUsage(err);
                    return EXIT_USAGE;
            }
        } catch (CliArgumentException e) {
            err.println(e.getMessage());
            err.println();
            printUsage(err);
            return EXIT_USAGE;
        } catch (IOException e) {
            err.println("Error: " + e.getMessage());
            return EXIT_ERROR;
        }
    }

    private static int cmdPull(String[] args, PrintStream out) throws IOException {
        StringBuilder registryPath = new StringBuilder();
        boolean[] offline = {false};
        StringBuilder modelsDir = new StringBuilder();
        String[] rest = stripPullOptions(args, registryPath, offline, modelsDir);
        if (rest.length < 1) {
            throw new CliArgumentException("pull requires a model spec: pull <url|name|alias|path>");
        }
        String spec = rest[0];
        ModelRegistry registry = openRegistry(orEmpty(registryPath));
        Path dir = modelsDir.length() > 0 ? Paths.get(modelsDir.toString()) : ModelPuller.DEFAULT_MODELS_DIR;
        ModelPuller puller =
                new ModelPuller(registry, new ModelNameResolver(), new HttpModelDownloader(), dir, offline[0]);
        ModelRegistryEntry entry = puller.pull(spec, (bytes, total) -> {
            if (total != null) {
                System.err.printf("\r%s  %d / %d bytes", spec, bytes, total);
            } else {
                System.err.printf("\r%s  %d bytes", spec, bytes);
            }
        });
        System.err.println();
        out.println("Pulled '" + entry.getName() + "' -> " + entry.getLocalPath());
        return EXIT_OK;
    }

    private static int cmdList(String[] args, PrintStream out) throws IOException {
        StringBuilder registryPath = new StringBuilder();
        String[] rest = stripOption(args, "--registry", registryPath);
        if (rest.length != 0) {
            throw new CliArgumentException("list takes no arguments");
        }
        ModelRegistry registry = openRegistry(orEmpty(registryPath));
        List<ModelRegistryEntry> entries = registry.list();
        if (entries.isEmpty()) {
            out.println("(no models registered)");
            return EXIT_OK;
        }
        for (ModelRegistryEntry e : entries) {
            out.println(e.getName() + "  quant=" + orDash(e.getQuantization()) + "  size=" + e.getSizeBytes()
                    + "  pulledAt=" + e.getPulledAt() + "  path=" + orDash(e.getLocalPath()));
            if (!e.getAliases().isEmpty()) {
                out.println("    aliases: " + String.join(", ", e.getAliases()));
            }
        }
        return EXIT_OK;
    }

    private static int cmdShow(String[] args, PrintStream out, PrintStream err) throws IOException {
        StringBuilder registryPath = new StringBuilder();
        String[] rest = stripOption(args, "--registry", registryPath);
        if (rest.length != 1) {
            throw new CliArgumentException("show requires a model name: show <name>");
        }
        ModelRegistry registry = openRegistry(orEmpty(registryPath));
        ModelRegistryEntry entry = registry.get(rest[0]);
        if (entry == null) {
            err.println("No such model: " + rest[1]);
            return EXIT_ERROR;
        }
        out.println(entry.toJsonNode().toString());
        return EXIT_OK;
    }

    private static int cmdRemove(String[] args, PrintStream out, PrintStream err) throws IOException {
        StringBuilder registryPath = new StringBuilder();
        String[] rest = stripOption(args, "--registry", registryPath);
        if (rest.length != 1) {
            throw new CliArgumentException("rm requires a model name: rm <name>");
        }
        ModelRegistry registry = openRegistry(orEmpty(registryPath));
        if (registry.remove(rest[0])) {
            out.println("Removed '" + rest[0] + "'");
            return EXIT_OK;
        }
        err.println("No such model: " + rest[0]);
        return EXIT_ERROR;
    }

    private static int cmdCopy(String[] args, PrintStream out, PrintStream err) throws IOException {
        StringBuilder registryPath = new StringBuilder();
        String[] rest = stripOption(args, "--registry", registryPath);
        if (rest.length != 2) {
            throw new CliArgumentException("cp requires src and dst: cp <src> <dst>");
        }
        String src = rest[0];
        String dst = rest[1];
        ModelRegistry registry = openRegistry(orEmpty(registryPath));
        ModelRegistryEntry source = registry.get(src);
        if (source == null) {
            err.println("No such model: " + src);
            return EXIT_ERROR;
        }
        if (registry.contains(dst)) {
            err.println("Destination already exists: " + dst);
            return EXIT_ERROR;
        }
        List<String> aliases = new ArrayList<>(source.getAliases());
        aliases.add(src);
        ModelRegistryEntry copy = new ModelRegistryEntry.Builder(dst)
                .localPath(source.getLocalPath())
                .sourceUrl(source.getSourceUrl())
                .quantization(source.getQuantization())
                .sizeBytes(source.getSizeBytes())
                .aliases(aliases)
                .pulledAt(System.currentTimeMillis())
                .build();
        registry.add(copy);
        out.println("Copied '" + src + "' -> '" + dst + "'");
        return EXIT_OK;
    }

    private static ModelRegistry openRegistry(String path) throws IOException {
        if (!path.isEmpty()) {
            return new ModelRegistry(Paths.get(path));
        }
        return ModelRegistry.load();
    }

    private static String[] stripOption(String[] args, String flag, StringBuilder captured) {
        List<String> rest = new ArrayList<>();
        for (int i = 1; i < args.length; i++) {
            if (flag.equals(args[i])) {
                if (i + 1 >= args.length) {
                    throw new CliArgumentException("Missing value for " + flag);
                }
                captured.append(args[++i]);
            } else {
                rest.add(args[i]);
            }
        }
        return rest.toArray(new String[0]);
    }

    private static String[] stripPullOptions(
            String[] args, StringBuilder registryPath, boolean[] offline, StringBuilder modelsDir) {
        List<String> rest = new ArrayList<>();
        for (int i = 1; i < args.length; i++) {
            String a = args[i];
            if ("--registry".equals(a)) {
                if (i + 1 >= args.length) {
                    throw new CliArgumentException("Missing value for --registry");
                }
                registryPath.append(args[++i]);
            } else if ("--offline".equals(a)) {
                offline[0] = true;
            } else if ("--models-dir".equals(a)) {
                if (i + 1 >= args.length) {
                    throw new CliArgumentException("Missing value for --models-dir");
                }
                modelsDir.append(args[++i]);
            } else {
                rest.add(a);
            }
        }
        return rest.toArray(new String[0]);
    }

    private static String orEmpty(StringBuilder sb) {
        return sb == null ? "" : sb.toString();
    }

    private static String orDash(@Nullable String s) {
        return s == null ? "-" : s;
    }

    private static void printUsage(PrintStream out) {
        out.println("jllama model registry - Ollama-like pull + manage");
        out.println();
        out.println("Usage:");
        out.println("  model-registry pull <url|name|alias|path> [--offline] [--models-dir <dir>]");
        out.println("  model-registry list");
        out.println("  model-registry show <name>");
        out.println("  model-registry rm <name>");
        out.println("  model-registry cp <src> <dst>");
        out.println();
        out.println("Global options:");
        out.println("  --registry <path>   Manifest file (default: ~/.jllama/models.json)");
    }

    /** Thrown for malformed CLI usage (missing/invalid arguments). */
    static final class CliArgumentException extends RuntimeException {
        CliArgumentException(String message) {
            super(message);
        }
    }
}
