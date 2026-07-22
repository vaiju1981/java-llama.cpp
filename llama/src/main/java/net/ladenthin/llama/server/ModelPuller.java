// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.jspecify.annotations.Nullable;

/**
 * Pulls models by name/URL/alias and records them in a {@link ModelRegistry} — the "Ollama-like"
 * {@code pull} verb. Resolution is delegated to a {@link ModelNameResolver}; the bytes are fetched by
 * a {@link ModelDownloader} (default: pure-Java {@link HttpModelDownloader}).
 *
 * <p>A {@code pull} of a remote spec refuses to run when {@link #isOffline() offline}, matching
 * {@code setOffline(true)} semantics. A {@code pull} of a local file path simply registers it without
 * touching the network. The registry manifest is persisted on every successful pull.
 */
public final class ModelPuller {

    /** Default model store: {@code <user.home>/.jllama/models}. */
    public static final Path DEFAULT_MODELS_DIR = Paths.get(System.getProperty("user.home", "."), ".jllama", "models");

    private final ModelRegistry registry;
    private final ModelNameResolver resolver;
    private final ModelDownloader downloader;
    private final Path modelsDir;
    private final boolean offline;

    /**
     * Construct a puller writing into {@link #DEFAULT_MODELS_DIR}, online, with default resolver and
     * downloader.
     *
     * @param registry the registry to record pulled models into
     */
    public ModelPuller(ModelRegistry registry) {
        this(registry, new ModelNameResolver(), new HttpModelDownloader(), DEFAULT_MODELS_DIR, false);
    }

    /**
     * Full constructor (used by tests and the CLI).
     *
     * @param registry the registry to record pulled models into
     * @param resolver the name/alias/URL resolver
     * @param downloader the transport that fetches remote bytes
     * @param modelsDir directory to write downloaded GGUF files into
     * @param offline when {@code true}, remote pulls are refused
     */
    public ModelPuller(
            ModelRegistry registry,
            ModelNameResolver resolver,
            ModelDownloader downloader,
            Path modelsDir,
            boolean offline) {
        this.registry = registry;
        this.resolver = resolver;
        this.downloader = downloader;
        this.modelsDir = modelsDir;
        this.offline = offline;
    }

    /**
     * Returns the model store directory.
     *
     * @return the model store directory
     */
    public Path getModelsDir() {
        return modelsDir;
    }

    /**
     * Returns whether remote pulls are refused.
     *
     * @return whether remote pulls are refused
     */
    public boolean isOffline() {
        return offline;
    }

    /**
     * Pull a spec and register the result. Convenience wrapper around {@link #pull(String,
     * PullProgressListener)} with no progress reporting.
     *
     * @param spec the URL, local path, alias, or {@code org/repo[@quant]} spec
     * @return the recorded registry entry
     * @throws IOException if resolution, download, or persistence fails
     */
    public ModelRegistryEntry pull(String spec) throws IOException {
        return pull(spec, null);
    }

    /**
     * Pull a spec and register the result.
     *
     * @param spec the URL, local path, alias, or {@code org/repo[@quant]} spec
     * @param listener optional progress callback (may be {@code null})
     * @return the recorded registry entry
     * @throws IOException if resolution, download, or persistence fails
     * @throws IllegalStateException if a remote pull is attempted while offline
     */
    public ModelRegistryEntry pull(String spec, @Nullable PullProgressListener listener) throws IOException {
        ResolvedModelSource src = resolver.resolve(spec);

        if (src.isLocal()) {
            String localPath = src.getLocalPath();
            if (localPath == null) {
                throw new IllegalStateException("local source has no path");
            }
            Path p = Paths.get(localPath);
            if (!Files.exists(p)) {
                throw new IOException("Local model not found: " + localPath);
            }
            ModelRegistryEntry entry = new ModelRegistryEntry.Builder(src.getName())
                    .localPath(p.toAbsolutePath().toString())
                    .quantization(src.getQuantization())
                    .sizeBytes(Files.size(p))
                    .pulledAt(System.currentTimeMillis())
                    .build();
            registry.add(entry);
            return entry;
        }

        if (offline) {
            throw new IllegalStateException("Offline mode: refusing to pull remote model '" + spec + "'");
        }

        Path downloaded = downloader.download(src, modelsDir, listener == null ? (b, t) -> {} : listener);
        ModelRegistryEntry entry = new ModelRegistryEntry.Builder(src.getName())
                .localPath(downloaded.toAbsolutePath().toString())
                .sourceUrl(src.getUrl())
                .quantization(src.getQuantization())
                .sizeBytes(Files.size(downloaded))
                .pulledAt(System.currentTimeMillis())
                .build();
        registry.add(entry);
        return entry;
    }
}
