// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import org.jspecify.annotations.Nullable;

/**
 * The result of resolving a pull spec (URL, alias, or local path) to a concrete source. A source is
 * either {@link #isRemote() remote} (has a {@code url} to download) or {@link #isLocal() local} (an
 * already-present file referenced by {@code localPath}).
 *
 * <p>Immutable. Auth headers (e.g. a HuggingFace token) travel here so the downloader stays
 * transport-agnostic.
 */
public final class ResolvedModelSource {

    private final String name;
    private final @Nullable String url;
    private final @Nullable String localPath;
    private final @Nullable String quantization;
    private final Map<String, String> headers;

    private ResolvedModelSource(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        this.url = b.url;
        this.localPath = b.localPath;
        this.quantization = b.quantization;
        this.headers = b.headers == null ? Collections.emptyMap() : Collections.unmodifiableMap(new LinkedHashMap<>(b.headers));
        if (url == null && localPath == null) {
            throw new IllegalStateException("a resolved source must have a url or a localPath");
        }
    }

    /**
     * Returns the stable model name.
     *
     * @return the stable model name (registry key)
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the download URL.
     *
     * @return the download URL, or {@code null} for a local source
     */
    public @Nullable String getUrl() {
        return url;
    }

    /**
     * Returns the local file path.
     *
     * @return the local file path, or {@code null} for a remote source
     */
    public @Nullable String getLocalPath() {
        return localPath;
    }

    /**
     * Returns the quantization label.
     *
     * @return the quantization label (e.g. {@code Q4_K_M}), or {@code null} when unknown
     */
    public @Nullable String getQuantization() {
        return quantization;
    }

    /**
     * Returns the extra request headers.
     *
     * @return extra request headers (e.g. {@code Authorization}), unmodifiable
     */
    public Map<String, String> getHeaders() {
        return headers;
    }

    /**
     * Returns whether this source is a local file.
     *
     * @return {@code true} if this source is a local file (no download needed)
     */
    public boolean isLocal() {
        return localPath != null;
    }

    /**
     * Returns whether this source must be downloaded.
     *
     * @return {@code true} if this source must be downloaded over the network
     */
    public boolean isRemote() {
        return url != null;
    }

    /** Builder for {@link ResolvedModelSource}. */
    public static final class Builder {
        private final String name;
        private @Nullable String url;
        private @Nullable String localPath;
        private @Nullable String quantization;
        private @Nullable Map<String, String> headers;

        /** @param name the stable model name (required, non-null) */
        public Builder(String name) {
            this.name = name;
        }

        public Builder url(@Nullable String url) {
            this.url = url;
            return this;
        }

        public Builder localPath(@Nullable String localPath) {
            this.localPath = localPath;
            return this;
        }

        public Builder quantization(@Nullable String quantization) {
            this.quantization = quantization;
            return this;
        }

        public Builder headers(@Nullable Map<String, String> headers) {
            this.headers = headers;
            return this;
        }

        public ResolvedModelSource build() {
            return new ResolvedModelSource(this);
        }
    }
}
