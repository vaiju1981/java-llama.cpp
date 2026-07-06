// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import lombok.ToString;
import org.jspecify.annotations.Nullable;

/**
 * Resolves library-specific system properties under the {@link #PREFIX} domain prefix.
 */
@ToString
public class LlamaSystemProperties {

    /** Creates a new {@link LlamaSystemProperties}. */
    public LlamaSystemProperties() {}

    /** Common system-property prefix for all library-specific overrides. */
    public static final String PREFIX = "net.ladenthin.llama";

    private @Nullable String getProperty(String suffix) {
        return System.getProperty(PREFIX + suffix);
    }

    /**
     * Custom directory containing the native jllama shared library.
     *
     * @return the configured library directory, or {@code null} if unset
     */
    public @Nullable String getLibPath() {
        return getProperty(".lib.path");
    }

    /**
     * Custom temporary directory used when extracting the native library from
     * the JAR. Falls back to {@code java.io.tmpdir} if absent.
     *
     * @return the configured temp directory, or {@code null} if unset
     */
    public @Nullable String getTmpDir() {
        return getProperty(".tmpdir");
    }

    /**
     * Architecture override for OS/arch detection in {@link OSInfo}.
     *
     * @return the configured architecture override, or {@code null} if unset
     */
    public @Nullable String getOsinfoArchitecture() {
        return getProperty(".osinfo.architecture");
    }

    /**
     * Number of GPU layers used in tests; parsed by the test suite.
     *
     * @return the configured GPU layer count as a string, or {@code null} if unset
     */
    public @Nullable String getTestNgl() {
        return getProperty(".test.ngl");
    }

    /**
     * Native-backend override for multi-backend ("all") fat jars that carry a
     * {@code jllama-backends.txt} manifest next to their native libraries. Names one backend
     * subdirectory (e.g. {@code cuda13}, {@code vulkan}) to load exclusively &mdash; loading then
     * fails loud instead of falling back &mdash; or the special value {@code default} (alias
     * {@code cpu}) to skip all manifest backends and load the default library directly. Ignored
     * by jars without a backend manifest.
     *
     * @return the configured backend name, or {@code null} if unset
     */
    public @Nullable String getBackend() {
        return getProperty(".backend");
    }
}
